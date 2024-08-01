/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
There's potentially a lot of elo available by adjusting the wdl
and lr schedulers, depending on your dataset.
*/
use bullet_lib::{
    inputs, lr, optimiser, outputs, wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule,
};

const HIDDEN_SIZE: usize = 2048;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .input(inputs::ChessBucketsMirroredFactorised::new([
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
             8,  9, 10, 11,
            12, 12, 13, 13,
            12, 12, 13, 13,
            14, 14, 15, 15,
            14, 14, 15, 15,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let sbs = 400;
    let schedule = TrainingSchedule {
        net_id: "lovecraft".into(),
        batch_size: 16_384,
        ft_regularisation: 1.0 / 16384.0 / 4194304.0,
        eval_scale: 400.0,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: sbs,
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.001 * 0.3 * 0.3 * 0.3,
                final_superbatch: sbs,
            },
            warmup_batches: 200,
        },
        loss_function: Loss::SigmoidMSE,
        save_rate: 20,
        optimiser_settings: optimiser::AdamWParams {
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.98,
            max_weight: 1.98,
        },
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["data/dataset.bin"],
        test_set: None,
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}