use crate::threat_inputs::ThreatInputs;
use acyclib::trainer::logger;
use bullet_lib::{
    game::{inputs::SparseInputType, outputs},
    nn::{Shape, optimiser},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

mod attacks;
mod indices;
mod offsets;
mod threat_inputs;
mod threats;

const STAGE1_SUPERBATCHES: usize = 100;
const STAGE2_SUPERBATCHES: usize = 800;
const STAGE3_SUPERBATCHES: usize = 200;

const L1_SIZE: usize = 640;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 32;

const OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
const KING_BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,
     8,  9, 10, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

fn main() {
    logger::set_cbcs(true);

    let inputs = ThreatInputs::new(KING_BUCKET_LAYOUT);

    let save_format = [
        SavedFormat::id("ftw"),
        SavedFormat::id("ftb"),
        SavedFormat::id("l1w"),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w"),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w"),
        SavedFormat::id("l3b"),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .output_buckets(outputs::MaterialCount::<OUTPUT_BUCKETS>)
        .optimiser(optimiser::Ranger)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            let ft = builder.new_affine("ft", inputs.num_inputs(), L1_SIZE);
            ft.init_with_effective_input_size(20000);

            let l1x = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2x = builder.new_affine("l2", L2_SIZE * 2, OUTPUT_BUCKETS * L3_SIZE);
            let l3x = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS);

            let stm_subnet = ft.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = ft.forward(ntm).crelu().pairwise_mul();
            let ft_out = stm_subnet.concat(ntm_subnet);

            let ones_l1_vec = builder.new_constant(Shape::new(1, L1_SIZE), &[1.0 / L1_SIZE as f32; L1_SIZE]);
            let ft_out_norm = ones_l1_vec.matmul(ft_out);

            let l1_out = l1x.forward(ft_out).select(buckets);
            let l1_out = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2x.forward(l1_out).select(buckets);
            let l2_out = l2_out.crelu();

            let l3_out = l3x.forward(l2_out).select(buckets);

            let loss = l3_out.sigmoid().squared_error(targets);
            let loss = loss + 0.005 * ft_out_norm;

            (l3_out, loss)
        });

    let net_id = std::env::args().nth(1).unwrap();

    let default_steps = TrainingSteps {
        batch_size: 16_384 * 8,
        batches_per_superbatch: 6104 / 8,
        start_superbatch: 1,
        end_superbatch: 0,
    };

    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            end_superbatch: STAGE1_SUPERBATCHES + STAGE2_SUPERBATCHES + STAGE3_SUPERBATCHES,
            ..default_steps
        },
        wdl_scheduler: wdl::Sequence {
            first: wdl::Sequence {
                first: wdl::ConstantWDL { value: 0.2 },
                second: wdl::LinearWDL { start: 0.4, end: 0.6 },
                first_scheduler_final_superbatch: STAGE1_SUPERBATCHES,
            },
            second: wdl::LinearWDL { start: 0.8, end: 0.9 },
            first_scheduler_final_superbatch: STAGE1_SUPERBATCHES + STAGE2_SUPERBATCHES,
        },
        lr_scheduler: lr::Sequence {
            first: lr::Sequence {
                first: lr::Warmup {
                    inner: lr::LinearDecayLR {
                        initial_lr: 0.001,
                        final_lr: 0.000027,
                        final_superbatch: STAGE1_SUPERBATCHES,
                    },
                    warmup_batches: 200,
                },
                second: lr::LinearDecayLR {
                    initial_lr: 0.001,
                    final_lr: 0.000027,
                    final_superbatch: STAGE2_SUPERBATCHES,
                },
                first_scheduler_final_superbatch: STAGE1_SUPERBATCHES,
            },
            second: lr::LinearDecayLR {
                initial_lr: 0.000025,
                final_lr: 0.0000025,
                final_superbatch: STAGE3_SUPERBATCHES,
            },
            first_scheduler_final_superbatch: STAGE1_SUPERBATCHES + STAGE2_SUPERBATCHES,
        },
        save_rate: 100,
    };

    let default_optimiser_params = optimiser::RangerParams {
        beta1: 0.99,
        beta2: 0.999,
        min_weight: -1.98,
        max_weight: 1.98,
        ..Default::default()
    };

    let ftw_optimiser_params =
        optimiser::RangerParams { min_weight: -0.99, max_weight: 0.99, ..default_optimiser_params };

    let l1w_clip = 0.99 * 255.0 * 255.0 / (256.0 * 256.0);
    //let l1w_clip = l1w_clip / 2.0;

    let l1w_optimiser_params =
        optimiser::RangerParams { min_weight: -l1w_clip, max_weight: l1w_clip, ..default_optimiser_params };

    trainer.optimiser.set_params(default_optimiser_params);

    //trainer.optimiser.set_params_for_weight("ftf", ftw_optimiser_params); // factoriser
    trainer.optimiser.set_params_for_weight("ftw", ftw_optimiser_params);

    //trainer.optimiser.set_params_for_weight("l1fw", l1w_optimiser_params);
    trainer.optimiser.set_params_for_weight("l1w", l1w_optimiser_params);

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let stage1_data_loader = loader::DirectSequentialDataLoader::new(&["/media/windows/NNUE/chess/038_040_043.bin"]);

    //trainer.load_from_checkpoint("checkpoints/net065-550");

    trainer.run(&schedule, &settings, &stage1_data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN:  {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
