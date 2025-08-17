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
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

const HL: usize = 2048;
const L2: usize = 16;
const L3: usize = 32;

const CLIP: f32 = 0.99 * 2.0;

const NUM_OUTPUT_BUCKETS: usize = 8;

fn main() {
    // hyperparams to fiddle with
    let dataset_path = "data/dataset.bin";
    let initial_lr = 0.001;
    let final_lr = 0.001 * f32::powi(0.3, 5);
    let superbatches = 800;
    let wdl_proportion = 0.4;
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

    let mut saves =
        ["l0w", "l0b", "l1w", "l1b" "l2w", "l2b", "l3w", "l3b"]
            .map(SavedFormat::id)
            .to_vec();

    // merge factoriser weights when saving:
    saves[0] = saves[0].clone().add_transform(|builder, _, mut weights| {
        let factoriser = builder.get_weights("l0f").get_dense_vals().unwrap();
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
            let l1 = builder.new_affine("l1", HL, NUM_OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2, NUM_OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    let adamw = AdamWParams { max_weight: CLIP, min_weight: -CLIP, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", adamw);
    trainer.optimiser.set_params_for_weight("l0f", adamw);
    trainer.optimiser.set_params_for_weight("l1w", adamw);
    trainer.optimiser.set_params_for_weight("l1b", adamw);
    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..adamw };
    trainer.optimiser.set_params_for_weight("l2w", no_clipping);
    trainer.optimiser.set_params_for_weight("l2b", no_clipping);
    trainer.optimiser.set_params_for_weight("l3w", no_clipping);
    trainer.optimiser.set_params_for_weight("l3b", no_clipping);

    let schedule = TrainingSchedule {
        net_id: "modern".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader = DirectSequentialDataLoader::new(&[dataset_path]);

    trainer.run(&schedule, &settings, &dataloader);
}
