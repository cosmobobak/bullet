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

const L1: usize = 2048;
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
    let dataset_path = "data/dataset.bin";
    let initial_lr = 0.001;
    let final_lr = 0.001 * f32::powi(0.3, 5);
    let superbatches = 800;
    let wdl_proportion = 0.4;

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

    let init_normal = |input: usize| InitSettings::Normal { mean: 0.0, stdev: (2.0 / (input as f32)).sqrt() };

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
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_l0f = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, L1);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_l0f).clip_pass_through_grad(-CLIP, CLIP);

            // layerstack weights + factorisers
            let mut l1 = builder.new_affine("l1x", L1, NUM_OUTPUT_BUCKETS * L2);
            let l1f = builder.new_weights("l1f", Shape::new(L2, L1), InitSettings::Zeroed);
            let mut l2 = builder.new_affine("l2x", L2, NUM_OUTPUT_BUCKETS * L3);
            let l2f = builder.new_weights("l2f", Shape::new(L3, L2), InitSettings::Zeroed);
            let mut l3 = builder.new_affine("l3x", L3, NUM_OUTPUT_BUCKETS);
            let l3f = builder.new_weights("l3f", Shape::new(1, L3), InitSettings::Zeroed);

            let expanded_l1f = l1f.repeat(NUM_OUTPUT_BUCKETS);
            l1.weights = (l1.weights + expanded_l1f).clip_pass_through_grad(-CLIP, CLIP);
            let expanded_l2f = l2f.repeat(NUM_OUTPUT_BUCKETS);
            l2.weights = l2.weights + expanded_l2f;
            let expanded_l3f = l3f.repeat(NUM_OUTPUT_BUCKETS);
            l3.weights = l3.weights + expanded_l3f;

            // inference
            let stm_subnet = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let accumulator = stm_subnet.concat(ntm_subnet);

            let l1_out = l1.forward(accumulator).select(output_buckets).screlu();
            let l2_out = l2.forward(l1_out).select(output_buckets).screlu();

            l3.forward(l2_out).select(output_buckets)
        });

    let adamw = AdamWParams { max_weight: 128.0, min_weight: -128.0, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", adamw);
    trainer.optimiser.set_params_for_weight("l0f", adamw);
    trainer.optimiser.set_params_for_weight("l1xw", adamw);
    trainer.optimiser.set_params_for_weight("l1xb", adamw);
    trainer.optimiser.set_params_for_weight("l1fw", adamw);
    trainer.optimiser.set_params_for_weight("l1fb", adamw);
    trainer.optimiser.set_params_for_weight("l2xw", adamw);
    trainer.optimiser.set_params_for_weight("l2xb", adamw);
    trainer.optimiser.set_params_for_weight("l2fw", adamw);
    trainer.optimiser.set_params_for_weight("l2fb", adamw);
    trainer.optimiser.set_params_for_weight("l3xw", adamw);
    trainer.optimiser.set_params_for_weight("l3xb", adamw);
    trainer.optimiser.set_params_for_weight("l3fw", adamw);
    trainer.optimiser.set_params_for_weight("l3fb", adamw);

    let schedule = TrainingSchedule {
        net_id: "tartarus".to_string(),
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
