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

const QA: i32 = 255;
const QB: i32 = 64;

fn main() {
    for hl_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096] {
        let mut trainer = TrainerBuilder::default()
            .optimiser(optimiser::AdamW)
            .quantisations(&[QA, QB])
            .input(inputs::Chess768)
            .output_buckets(outputs::Single)
            .feature_transformer(hl_size)
            .activate(Activation::SCReLU)
            .add_layer(1)
            .build();

        let schedule = TrainingSchedule {
            net_id: format!("optimiser-benchmark-screlu-warmupcosinedecay-{hl_size}n"),
            eval_scale: 400.0,
            ft_regularisation: 0.0,
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 10,
            wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
            lr_scheduler: lr::Warmup { inner: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.1 * 0.1, final_superbatch: 10 }, warmup_batches: 256 },
            loss_function: Loss::SigmoidMSE,
            save_rate: 100,
            optimiser_settings: optimiser::AdamWParams { decay: 0.01 },
        };

        let settings =
            LocalSettings { threads: 4, data_file_paths: vec!["data/input.bin"], output_directory: "checkpoints" };

        trainer.run(&schedule, &settings);
    }
}
