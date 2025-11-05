use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};

const HL: usize = 2048;
const L2: usize = 16;
const L3: usize = 32;

const CLIP: f32 = 0.99 * 2.0;

const NUM_OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,
     8,  9, 10, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn main() {
    // hyperparams to fiddle with
    let mut saves =
        ["l0w", "l0b", "l1xw", "l1fw", "l1xb", "l1fb", "l2xw", "l2fw", "l2xb", "l2fb", "l3xw", "l3fw", "l3xb", "l3fb"]
            .map(SavedFormat::id)
            .to_vec();

    // merge factoriser weights when saving:
    saves[0] = saves[0].clone().transform(|builder, mut weights| {
        let factoriser = builder.get("l0f").values;
        let expanded = factoriser.repeat(weights.len() / factoriser.len());

        for (i, j) in weights.iter_mut().zip(expanded.iter()) {
            *i += *j;
        }

        weights
    });

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&saves)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // builder.dump_graphviz("viz.txt");
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HL, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HL);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            // layerstack weights
            let l1x = builder.new_affine("l1x", HL, NUM_OUTPUT_BUCKETS * L2);
            let l1f = builder.new_affine("l1f", HL, L2);
            let l2x = builder.new_affine("l2x", L2, NUM_OUTPUT_BUCKETS * L3);
            let l2f = builder.new_affine("l2f", L2, L3);
            let l3x = builder.new_affine("l3x", L3, NUM_OUTPUT_BUCKETS);
            let l3f = builder.new_affine("l3f", L3, 1);

            // inference
            let stm_subnet = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let accumulator = stm_subnet.concat(ntm_subnet);

            let l1x_out = l1x.forward(accumulator).select(output_buckets);
            let l1f_out = l1f.forward(accumulator);
            let l1_out = (l1x_out + l1f_out).screlu();

            let l2x_out = l2x.forward(l1_out).select(output_buckets);
            let l2f_out = l2f.forward(l1_out);
            let l2_out = (l2x_out + l2f_out).screlu();

            let l3x_out = l3x.forward(l2_out).select(output_buckets);
            let l3f_out = l3f.forward(l2_out);
            let l3_out = l3x_out + l3f_out;

            l3_out
        });

    let adamw = AdamWParams { max_weight: CLIP, min_weight: -CLIP, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", adamw);
    trainer.optimiser.set_params_for_weight("l0f", adamw);
    trainer.optimiser.set_params_for_weight("l1xw", adamw);
    trainer.optimiser.set_params_for_weight("l1xb", adamw);
    trainer.optimiser.set_params_for_weight("l1fw", adamw);
    trainer.optimiser.set_params_for_weight("l1fb", adamw);
    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..adamw };
    trainer.optimiser.set_params_for_weight("l2xw", no_clipping);
    trainer.optimiser.set_params_for_weight("l2xb", no_clipping);
    trainer.optimiser.set_params_for_weight("l2fw", no_clipping);
    trainer.optimiser.set_params_for_weight("l2fb", no_clipping);
    trainer.optimiser.set_params_for_weight("l3xw", no_clipping);
    trainer.optimiser.set_params_for_weight("l3xb", no_clipping);
    trainer.optimiser.set_params_for_weight("l3fw", no_clipping);
    trainer.optimiser.set_params_for_weight("l3fb", no_clipping);

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let filter = viriformat::dataformat::Filter {
        min_ply: 16,
        min_pieces: 4,
        max_eval: 20_000,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,

        // from Default::default()
        random_fen_skipping: true,
        random_fen_skip_probability: 9.0 / 10.0,
        wdl_filtered: false,
        wdl_model_params_a: [6.871_558_62, -39.652_263_91, 90.684_603_52, 170.669_963_64],
        wdl_model_params_b: [-7.198_907_10, 56.139_471_85, -139.910_911_83, 182.810_074_27],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.5,
    };

    // trainer.load_from_checkpoint("checkpoints/haruspex-800");

    let schedule1 = TrainingSchedule {
        net_id: "logos".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 800,
        },
        wdl_scheduler: wdl::LinearWDL { start: 0.15, end: 0.6 },
        lr_scheduler: lr::Warmup {
            inner: lr::Sequence {
                first: lr::ConstantLR { value: 0.001 },
                first_scheduler_final_superbatch: 600,
                second: lr::CosineDecayLR {
                    initial_lr: 0.001,
                    final_lr: 0.001 * 0.3 * 0.3 * 0.3,
                    final_superbatch: 200,
                },
            },
            warmup_batches: 1600,
        },
        save_rate: 400,
    };

    let dataloader1 = bullet_lib::value::loader::ViriBinpackLoader::new("data/all.vf", 4096, 16, filter.clone());

    trainer.run(&schedule1, &settings, &dataloader1);

    drop(dataloader1);

    let schedule2 = TrainingSchedule {
        net_id: "logos-finetuned".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 800,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.000025,
            final_lr: 0.000025 * 0.3 * 0.3 * 0.3,
            final_superbatch: 400,
        },
        save_rate: 400,
    };

    let dataloader2 = bullet_lib::value::loader::ViriBinpackLoader::new("data/2025-01-forward.vf", 4096, 16, filter);

    trainer.run(&schedule2, &settings, &dataloader2);
}
