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

const HL: usize = 2560;
const L2: usize = 16;
const L3: usize = 32;
const HEADS: usize = 1;

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
    let dataset_path = "data/all-relabelled-v2.vf";
    let initial_lr = 0.001;
    let superbatches = 800;
    let lr_scheduler = lr::Warmup {
        inner: lr::CosineDecayLR {
            initial_lr,
            final_lr: initial_lr * f32::powi(0.3, 5),
            final_superbatch: superbatches,
        },
        warmup_batches: 1600,
    };
    let wdl_scheduler = wdl::LinearWDL { start: 0.4, end: 1.0 };

    let mut saves = ["l0w", "l0b", "l1w", "l1b", "l2xw", "l2fw", "l2xb", "l2fb", "l3xw", "l3fw", "l3xb", "l3fb"]
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
        // .wdl_output()
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .dual_perspective()
        .optimiser(AdamW)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&saves)
        // .wdl_adjust_function(|pos, _wdl| {
        //     let mut antiphase = 0;
        //     for (piece, _) in pos.into_iter() {
        //         antiphase += match piece & 7 {
        //             0 => 1,
        //             1 | 2 => 4,
        //             3 => 6,
        //             4 => 12,
        //             5 => 0,
        //             _ => panic!(),
        //         };
        //     }
        //     let phase = 96 - i32::min(96, antiphase);
        //     let phase = phase as f32 / 96.0;
        //     // phase * phase   ← what wonders lie in distant past, hidden in the towers of witches
        //     0.4 + 0.6 * phase
        // })
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), targets| {
            // builder.dump_graphviz("viz.txt");
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HL, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HL);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // layerstack weights
            let l1 = builder.new_affine("l1", HL, NUM_OUTPUT_BUCKETS * L2);
            let l2x = builder.new_affine("l2x", L2, NUM_OUTPUT_BUCKETS * L3);
            let l2f = builder.new_affine("l2f", L2, L3);
            let l3x = builder.new_affine("l3x", L3, NUM_OUTPUT_BUCKETS * HEADS);
            let l3f = builder.new_affine("l3f", L3, HEADS);

            // inference
            let stm_subnet = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let accumulator = stm_subnet.concat(ntm_subnet);

            let l1_out = l1.forward(accumulator).select(output_buckets).screlu();

            let l2x_out = l2x.forward(l1_out).select(output_buckets);
            let l2f_out = l2f.forward(l1_out);
            let l2_out = (l2x_out + l2f_out).screlu();

            let l3x_out = l3x.forward(l2_out).select(output_buckets);
            let l3f_out = l3f.forward(l2_out);

            let out = l3x_out + l3f_out;

            // let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
            // let loss = ones.matmul(out.softmax_crossentropy_loss(targets));

            let loss = out.sigmoid().squared_error(targets);

            (out, loss)
        });

    let adamw = AdamWParams { max_weight: CLIP, min_weight: -CLIP, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", adamw);
    trainer.optimiser.set_params_for_weight("l1b", adamw);
    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..adamw };
    trainer.optimiser.set_params_for_weight("l0w", no_clipping);
    trainer.optimiser.set_params_for_weight("l0f", no_clipping);
    trainer.optimiser.set_params_for_weight("l2xw", no_clipping);
    trainer.optimiser.set_params_for_weight("l2xb", no_clipping);
    trainer.optimiser.set_params_for_weight("l2fw", no_clipping);
    trainer.optimiser.set_params_for_weight("l2fb", no_clipping);
    trainer.optimiser.set_params_for_weight("l3xw", no_clipping);
    trainer.optimiser.set_params_for_weight("l3xb", no_clipping);
    trainer.optimiser.set_params_for_weight("l3fw", no_clipping);
    trainer.optimiser.set_params_for_weight("l3fb", no_clipping);

    let schedule = TrainingSchedule {
        net_id: "equilibrium".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler,
        lr_scheduler,
        save_rate: 10000,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    // let dataloader = bullet_lib::value::loader::DirectSequentialDataLoader::new(&[dataset_path]);
    let dataloader = bullet_lib::value::loader::ViriBinpackLoader::new(
        dataset_path,
        4096,
        16,
        viriformat::dataformat::Filter {
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
        },
    );

    // trainer.load_from_checkpoint("checkpoints/hapax-800");

    trainer.run(&schedule, &settings, &dataloader);
}
