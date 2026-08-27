use std::cell::RefCell;

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType as _, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, ModelBuilder, ModelNode, Shape,
        optimiser::{Optimiser, RangerOptimiser, RangerParams},
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{
        loader::ViriBinpackLoader,
        save::{save_to_checkpoint, write_losses},
    },
};
use bullet_trainer::{
    model::{ModelDefinition, ModelEvaluator, ModelInputs, ModelWeights, SavedFormat},
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, logger, train},
};

use crate::tipp_inputs::TiPpInputs;

mod tipp_inputs;

const NET_ID: &str = "hyperion";

const SEED: u64 = 42;

const L1: usize = 1024;
const D: usize = 32;
const PROJ: usize = 1;
const HEADS: usize = 1;

// weight of the auxiliary WDL-classification cross-entropy loss
// const WDL_CE_ALPHA: f32 = 0.02;

// penalty on WDL logits
// const WDL_Z_BETA: f32 = 5e-6;

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

const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

const Q0: i16 = 255;
const Q1: i16 = 128;

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;
const TIPP_RANGE: f32 = i8::MAX as f32 / (Q0 as f32);

const BATCH_GLOM: usize = 4;

// we could set this lower for SWA,
// but tests indicate it doesn’t really help.
// https://viri.dev/test/694/
//   LLR −2.95 (−2.94 LO +2.94 HI BND for +0.00 LO +3.00 HI ELO)
//  PERF −0.77 ± 1.74 ELO = −2.51 LO +0.96 HI ELO
//  CONF 8+0.08 SEC 1 THREAD 16 MB CACHE
// GAMES 39056 = 9744W 19481D 9831L = 24.9%W 49.9%D 25.2%L
// PENTA 85⁺² 4586⁺¹ 10086⁰ 4699⁻¹ 72⁻²
const SAVE_RATE: usize = 10000;

// values verbatim from a pawnocchio schedule
const SUPERBATCHES_STAGE0: usize = 100;
const SUPERBATCHES_STAGE1: usize = 800;
const SUPERBATCHES_STAGE2: usize = 200;

fn main() {
    let tipp = TiPpInputs::new(tipp_inputs::three_file_band_mask());
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let output_buckets = MaterialCount::<NUM_OUTPUT_BUCKETS>;

    // hyperparams to fiddle with
    let dataset_path = "data/all-relabelled.vf";

    let mut saves = vec![
        SavedFormat::id("l0tippw"),
        SavedFormat::id("l0psqt").transform(|weights, values| {
            let fac = weights.get("l0fac").values.f32().repeat(INPUT_BUCKETS);
            assert_eq!(values.len(), fac.len());
            values.iter().zip(fac).map(|(&a, b)| a + b).collect()
        }),
    ];

    saves.extend(
        [
            "l0tippb", "l1w", "l1b", // "l1n_g", "l1n_b",
            "l2up_xw", "l2up_fw", "l2up_xb", "l2up_fb",
            // "l2down_xw",
            // "l2down_fw",
            // "l2down_xb",
            // "l2down_fb",
            "l3xw", "l3fw", "l3xb", "l3fb",
        ]
        .map(SavedFormat::id),
    );

    let inputs = ModelInputs::default()
        .add_sparse("stm/tipp", (tipp.num_inputs(), 1), tipp.max_active())
        .add_sparse("ntm/tipp", (tipp.num_inputs(), 1), tipp.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("buckets", (NUM_OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (4, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((stm_tipp, ntm_tipp), stm_psqt), ntm_psqt), buckets), targets)| {
            // input layer factoriser
            let l0tipp = builder.new_affine("l0tipp", tipp.num_inputs(), L1);

            let l0fac = builder.new_weights("l0fac", Shape::new(L1, 768), InitSettings::Zeroed);
            let psqt_init = InitSettings::Normal { mean: 0.0, stdev: 2.0 / 32f32.sqrt() };
            let mut l0psqt = builder.new_weights("l0psqt", Shape::new(L1, psqt.num_inputs()), psqt_init);
            l0psqt = l0psqt + l0fac.repeat(psqt.num_inputs() / 768);

            // layerstack weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * D);
            let l2up_x = builder.new_affine("l2up_x", D, NUM_OUTPUT_BUCKETS * D * PROJ * 2);
            let l2up_f = builder.new_affine("l2up_f", D, D * PROJ * 2);
            // let l2down_x = builder.new_affine("l2down_x", D * PROJ, NUM_OUTPUT_BUCKETS * D);
            // let l2down_f = builder.new_affine("l2down_f", D * PROJ, D);
            let l3x = builder.new_affine("l3x", D, NUM_OUTPUT_BUCKETS * HEADS);
            let l3f = builder.new_affine("l3f", D, HEADS);
            // auxiliary WDL-classification head, training-only (not saved)
            // let l3wdl_x = builder.new_affine("l3wdl_x", D, NUM_OUTPUT_BUCKETS * 3);
            // let l3wdl_f = builder.new_affine("l3wdl_f", D, 3);

            // inference
            let ft = |tipp, psqt, start, end| {
                (l0tipp.slice(start, end).forward(tipp) + l0psqt.slice_rows(start, end).matmul(psqt)).crelu()
            };
            let stm_subnet = ft(stm_tipp, stm_psqt, 0, L1 / 2) * ft(stm_tipp, stm_psqt, L1 / 2, L1);
            let ntm_subnet = ft(ntm_tipp, ntm_psqt, 0, L1 / 2) * ft(ntm_tipp, ntm_psqt, L1 / 2, L1);
            let l0_out = stm_subnet.concat(ntm_subnet);

            // L₁-norm penalty on accumulator (mean, since values are non-negative):
            let mean_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = mean_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(buckets);
            let l1_out = hard_swish(l1_out);

            // let l1n_out = rms_norm(builder, "l1n", l1_out);
            let l1n_out = l1_out; // todo: test norm.

            // up-projection:
            let l2x_proj = l2up_x.forward(l1n_out).select(buckets);
            let l2f_proj = l2up_f.forward(l1n_out);
            let l2_proj = l2x_proj + l2f_proj;
            // activation:
            let l2_proj_gate = hard_swish(l2_proj.slice_rows(0, D * PROJ));
            let l2_proj_id = l2_proj.slice_rows(D * PROJ, D * PROJ * 2);
            let l2_proj = l2_proj_gate * l2_proj_id;
            // down-projection:
            // let l2x_out = l2down_x.forward(l2_proj).select(buckets);
            // let l2f_out = l2down_f.forward(l2_proj);
            // let l2_out = l2x_out + l2f_out;
            let l2_out = l2_proj;

            // skip connexion from l1-out to l2-out:
            let l2_out = l2_out + l1_out;

            let l3x_out = l3x.forward(l2_out).select(buckets);
            let l3f_out = l3f.forward(l2_out);

            let l3_out = l3x_out + l3f_out;

            let loss = if HEADS == 3 {
                // -------- MSE --------
                let loss_mask = builder.new_constant(Shape::new(1, 3), &[1.0, 0.0, 0.0]);
                let draw_mask = builder.new_constant(Shape::new(1, 3), &[0.0, 1.0, 0.0]);
                let win_mask = builder.new_constant(Shape::new(1, 3), &[0.0, 0.0, 1.0]);

                let loss = loss_mask.matmul(l3_out);
                let draw = draw_mask.matmul(l3_out);
                let win = win_mask.matmul(l3_out);

                let max = maximum(loss, maximum(draw, win));

                let loss = exp(loss - max);
                let draw = exp(draw - max);
                let win = exp(win - max);

                let inv_sum = (win + draw + loss).abs_pow(-1.0);
                let win = win * inv_sum;
                let draw = draw * inv_sum;

                // Calculate score from target
                let target_value = targets.slice_rows(0, 1);
                let targets = targets.slice_rows(1, 4);

                // Calculate MSE loss
                let mse_result = (draw * 0.5 + win).crelu(); // .clamp(0.0, 1.0)
                let mse_loss = mse_result.squared_error(target_value);

                // -------- CE --------
                let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
                let ce_loss = ones.matmul(l3_out.softmax_crossentropy_loss(targets));

                let loss = mse_loss + 0.1 * ce_loss;

                loss + 0.005 * l0_out_norm
            } else {
                // targets: row 0 is the WDL-blended value, rows 1..4 one-hot game result
                let target_value = targets.slice_rows(0, 1);
                // let target_wdl = targets.slice_rows(1, 4);

                let value_loss = l3_out.sigmoid().squared_error(target_value);

                // let wdl_logits = l3wdl_x.forward(l2_out).select(buckets) + l3wdl_f.forward(l2_out);
                // let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
                // let wdl_loss = ones.matmul(wdl_logits.softmax_crossentropy_loss(target_wdl));
                // let wdl_logit_norm = ones.matmul(wdl_logits * wdl_logits);

                value_loss + 0.005 * l0_out_norm // + WDL_CE_ALPHA * wdl_loss + WDL_Z_BETA * wdl_logit_norm;
            };

            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), l3_out)])
        },
    );

    let default_optimiser_params =
        RangerParams { beta1: 0.99, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, ..Default::default() };

    let weights = ModelWeights::new(&defn, SEED);
    let device = DefaultDevice::new(0).unwrap();

    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser =
        Optimiser::<_, RangerOptimiser>::new(defn, weights, device.clone(), default_optimiser_params).unwrap();

    optimiser.set_params(default_optimiser_params);

    let l0_clip = RangerParams { min_weight: -0.99, max_weight: 0.99, ..default_optimiser_params };
    optimiser.set_params_for_weight("l0fac", l0_clip);
    optimiser.set_params_for_weight("l0psqt", l0_clip);

    let tipp_clip = RangerParams { min_weight: -TIPP_RANGE, max_weight: TIPP_RANGE, ..default_optimiser_params };
    optimiser.set_params_for_weight("l0tippw", tipp_clip);

    let l1_clip = RangerParams { min_weight: -L1_RANGE, max_weight: L1_RANGE, ..default_optimiser_params };
    optimiser.set_params_for_weight("l1w", l1_clip);

    // don't bother clipping the float layers
    let no_clipping = RangerParams { min_weight: -128.0, max_weight: 128.0, ..default_optimiser_params };
    for name in [
        // "l1n_g",
        // "l1n_b",
        "l2up_xw", "l2up_xb", "l2up_fw", "l2up_fb",
        // "l2down_xw",
        // "l2down_xb",
        // "l2down_fw",
        // "l2down_fb",
        "l3xw", "l3xb", "l3fw", "l3fb",
        // "l3wdl_xw", "l3wdl_xb", "l3wdl_fw", "l3wdl_fb",
    ] {
        optimiser.set_params_for_weight(name, no_clipping);
    }

    let dataloader = ViriBinpackLoader::new(
        dataset_path,
        4096,
        16,
        viriformat::dataformat::Filter {
            max_eval: 20_000,
            random_fen_skipping: true,
            random_fen_skip_probability: 9.0 / 10.0,
            ..Default::default()
        },
    );

    let params = (&inputs, &tipp, psqt, output_buckets);

    let _ = std::fs::create_dir("checkpoints");

    let mut run = |stage, end_superbatch, lr_schedule, mapper| {
        let net_id = format!("{NET_ID}-s{stage}");

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        let steps = TrainingSteps {
            batch_size: 16_384 * BATCH_GLOM,
            batches_per_superbatch: 6104 / BATCH_GLOM,
            start_superbatch: 1,
            end_superbatch,
        };

        steps.display();

        train(
            &mut optimiser,
            TrainingSchedule { steps, lr_schedule, log_rate: 128 },
            ReadMapLoader::new(dataloader.clone(), mapper, 4),
            |_, step, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if step.batch().is_multiple_of(32)
                    || (step.batches_per_superbatch() < 32 && step.batch() == step.batches_per_superbatch())
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, step.batches_per_superbatch() as f32);

                    error_record.borrow_mut().push((step.superbatch(), step.batch(), normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |optimiser, step| {
                let superbatch = step.superbatch();
                if superbatch % SAVE_RATE == 0 || superbatch == step.final_superbatch() {
                    let name = format!("{net_id}-{superbatch}");
                    let path = format!("checkpoints/{name}");
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save_to_checkpoint(optimiser, &saves, &path);
                    write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));
                }
            },
        )
        .unwrap();
    };

    const WARMUP_SBS: usize = SUPERBATCHES_STAGE0 / 2;
    const COOLDOWN_SBS: usize = SUPERBATCHES_STAGE0 - WARMUP_SBS;
    run(
        0,
        SUPERBATCHES_STAGE0,
        lr::Sequence {
            first: lr::LinearDecayLR { initial_lr: 1e-4, final_lr: 5e-3, final_superbatch: WARMUP_SBS },
            second: lr::LinearDecayLR { initial_lr: 5e-3, final_lr: 1e-4, final_superbatch: COOLDOWN_SBS },
            first_scheduler_final_superbatch: WARMUP_SBS,
        }
        .boxed(),
        tipp_inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.2 }),
    );

    run(
        1,
        SUPERBATCHES_STAGE1,
        lr::LinearDecayLR { initial_lr: 1e-3, final_lr: 1e-6, final_superbatch: SUPERBATCHES_STAGE1 }.boxed(),
        tipp_inputs::make_inputs_mapper(params, wdl::LinearWDL { start: 0.2, end: 0.5 }),
    );

    run(
        2,
        SUPERBATCHES_STAGE2,
        lr::LinearDecayLR { initial_lr: 1e-5, final_lr: 1e-7, final_superbatch: SUPERBATCHES_STAGE2 }.boxed(),
        tipp_inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 1.0 }),
    );

    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let evaluator_mapper = tipp_inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.0 });

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
        "3N4/b2R2p1/3q3r/6P1/4k1nQ/7B/8/K7 w - - 0 1",
        "k2B1Q1q/8/b7/4p3/3Pr3/1N5R/2n5/1K6 w - - 0 1",
        "1B3q2/8/r5n1/8/Rp1N1PQ1/8/4bk2/2K5 w - - 0 1",
        "8/5NR1/5q1b/8/7p/3P2B1/6Q1/1k1K1n1r w - - 0 1",
        "8/8/6r1/4B3/3Q3p/N1nq4/5RP1/b3K2k b - - 0 1",
        "3qn2Q/1R6/8/1N3b1p/4B3/1kP5/r7/5K2 b - - 0 1",
        "3rBR2/2qQ1p2/N7/2P2b2/6n1/k7/8/6K1 b - - 0 1",
        "k7/8/p1rB1q2/7Q/4R3/2N2n2/7P/6bK b - - 0 1",
        "2n2Rr1/Bk5p/N7/2Q3q1/b7/8/KP6/8 w - - 0 1",
        "8/Q6r/3qR1P1/b4p2/k7/3B4/1KN2n2/8 b - - 0 1",
        "2nR4/1qB5/2p5/7r/4bQ2/1P1N4/2K1k3/8 w - - 0 1",
        "8/2Q1B3/n3qR1r/bk1p4/1P6/8/3K4/7N w - - 0 1",
        "7r/4b3/4k1N1/2q4n/1Q2B3/R5p1/1P2K3/8 b - - 0 1",
        "2r1n1k1/NbR5/6B1/2p1P3/8/8/5K2/q6Q b - - 0 1",
        "2Q2R2/P1pn4/q1N5/1b5k/1r6/B7/6K1/8 b - - 0 1",
        "1Nr2b2/R1p5/5q2/7B/2P5/3nk3/7K/1Q6 w - - 0 1",
        "4Q3/6P1/1k3p2/4N3/2r5/K6b/1n1B2Rq/8 b - - 0 1",
        "1B5Q/1n6/2p1rN2/3R4/3P4/1K3k2/3b4/6q1 w - - 0 1",
        "3n4/3q4/5Q2/4rP2/1N2p3/2K2B2/5k2/2b4R b - - 0 1",
        "6B1/2k5/2n1R3/1q2p3/2P4Q/3K4/r5b1/3N4 w - - 0 1",
        "8/8/b6N/R3pr1n/Q7/1Pk1K3/4B3/5q2 b - - 0 1",
        "3Q2r1/4P2R/1b6/8/8/1B3K2/4p2q/1k1n1N2 w - - 0 1",
        "bR5q/2r3B1/2Q1P3/8/2n5/1N1p2K1/k7/8 w - - 0 1",
        "1q1b2r1/8/8/2p5/4N3/3k1P1K/2nB1Q2/4R3 w - - 0 1",
        "5rRq/8/1Qn5/8/K7/P1B4b/1p2N3/7k w - - 0 1",
        "1n6/8/B3q3/5R2/1KPb2N1/7Q/r4p2/2k5 w - - 0 1",
        "q3N1R1/8/1B5n/2p5/2K2P2/7r/1b1k4/7Q w - - 0 1",
        "1B6/N6q/2b5/7R/P2K4/1Q1pr3/6n1/2k5 b - - 0 1",
        "1R3q2/p3Q1n1/4N3/6r1/4K1B1/2P5/7b/4k3 w - - 0 1",
        "1k6/2RQP3/1p6/b7/1B3K2/r1n5/3Nq3/8 b - - 0 1",
        "b7/k7/5P2/n2N4/5pK1/2q5/2B2R2/r4Q2 b - - 0 1",
        "1B6/P4q2/5r2/8/1k2n2K/5b2/1NR1p3/6Q1 w - - 0 1",
        "8/3Pk3/B2r4/K5N1/b7/3n1p1Q/2R5/5q2 w - - 0 1",
        "q2Q1R2/2p4N/1b1P4/1K6/1B3r2/8/8/n2k4 w - - 0 1",
        "n1k5/5pq1/R4b2/2K5/3N4/7P/4BrQ1/8 b - - 0 1",
        "8/4Q3/B7/3KN1P1/3b4/nk3p2/8/R4r1q w - - 0 1",
        "b6n/B1k5/8/4KN1r/1Q6/7R/6Pp/5q2 b - - 0 1",
        "6k1/7r/8/bB3K1N/1R1q4/4Q3/2nP1p2/8 w - - 0 1",
        "Q6R/8/2B1q3/3N1nK1/2kb4/P7/r6p/8 w - - 0 1",
        "8/p5r1/k7/6PK/3b4/2B5/n4qQ1/3N2R1 w - - 0 1",
        "4kb2/6r1/K7/p7/6n1/2N5/2BP1qR1/7Q w - - 0 1",
        "6q1/1BN5/1K3P2/3br1np/3R4/Q7/8/5k2 w - - 0 1",
        "5n2/5q2/1NK5/k1P3r1/3p4/7Q/B6b/1R6 w - - 0 1",
        "B3r3/3p4/N2K2k1/1Q6/2R5/1bP5/1q5n/8 w - - 0 1",
        "BR2Q3/4N3/1n2K3/k7/1p1b1q2/8/5P2/7r b - - 0 1",
        "1k6/7R/5K1N/1pQ5/1n6/P4b2/1r6/6qB b - - 0 1",
        "8/3k4/3NnPK1/3QR3/3r2pB/8/4b3/q7 w - - 0 1",
        "1Q6/4q3/NB5K/1R1r4/3P4/bp1k4/6n1/8 w - - 0 1",
        "3Br3/K7/2q1N3/7n/8/4PbRQ/1p1k4/8 w - - 0 1",
        "R2r4/pK1b4/1n4NB/7P/8/3Q4/6k1/4q3 b - - 0 1",
        "3N2r1/2KP4/8/1B1p4/2b5/3RQq2/2k5/7n w - - 0 1",
        "5q2/1N1KB3/5b2/p4R2/4k3/P7/Q7/4n1r1 b - - 0 1",
        "NR6/4K3/1q3r2/3Q3P/3n2k1/8/7p/B5b1 b - - 0 1",
        "q7/1N1B1K2/1Q6/5b2/5pP1/6r1/n6k/R7 w - - 0 1",
        "2R5/2n1k1K1/5r2/3P4/2Q4p/2q5/6NB/7b w - - 0 1",
        "3n1Qr1/3p3K/8/3B4/R5b1/4P3/1qN4k/8 w - - 0 1",
        "K7/3k4/3n2b1/1P2r3/8/p2Bq3/3R4/3QN3 b - - 0 1",
        "1K6/8/3rRN2/1BP3b1/3p4/8/k2n2q1/5Q2 w - - 0 1",
        "2K5/6Bn/p4r2/2P1Q3/1qb5/8/2R5/3kN3 w - - 0 1",
        "3K4/8/2bP4/1qN5/2n3B1/3R4/4Qrp1/6k1 b - - 0 1",
        "1B2K1k1/P3b3/5q2/3R4/1pQ2r1n/8/8/6N1 b - - 0 1",
        "5K2/p4P1b/5QB1/4q3/6k1/8/4r3/R1n1N3 b - - 0 1",
        "6K1/8/b6R/N2p2P1/8/q1Q5/6r1/2Bk3n b - - 0 1",
        "7K/r2R3b/1Q6/8/2q5/1nPB2k1/N3p3/8 w - - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let inputs = evaluator_mapper.map(&[pos], Default::default(), 1).to_device(&device).unwrap();
        let output = evaluator.evaluate(&inputs).unwrap().get("output").unwrap();
        let [value] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * value);
    }
}

fn maximum<'a>(x: ModelNode<'a>, y: ModelNode<'a>) -> ModelNode<'a> {
    (x - y).relu() + y
}

// computes e^x via 1 / (1/σ(x) - 1), since 1/σ(x) - 1 = e^(-x)
fn exp(x: ModelNode) -> ModelNode {
    let sigmoid = x.sigmoid();
    let inv_sigmoid = sigmoid.abs_pow(-1.0);
    let e_minus_x = inv_sigmoid - 1.0;
    e_minus_x.abs_pow(-1.0)
}

fn hard_swish(x: ModelNode) -> ModelNode {
    let gate = (x * (1.0 / 6.0) + 0.5).crelu();
    x * gate
}

fn _broadcast_rows<'a>(builder: &'a ModelBuilder, scalar: ModelNode<'a>, rows: usize) -> ModelNode<'a> {
    let ones = builder.new_constant(Shape::new(rows, 1), &vec![1.0; rows]);
    ones.matmul(scalar)
}

fn _rms_norm<'a>(builder: &'a ModelBuilder, id: &str, x: ModelNode<'a>) -> ModelNode<'a> {
    const EPS: f32 = 1e-5;
    let n = x.shape().rows();
    assert_eq!(x.shape().cols(), 1, "rms_norm expects a column vector");

    // mean of squares over the feature dimension (rows), broadcast back to (n, 1)
    let mean_sq = (x * x).reduce_sum_rows() / n as f32;
    let inv_rms = _broadcast_rows(builder, (mean_sq + EPS).abs_pow(-0.5), n);
    let normed = x * inv_rms;

    // γ as a 0-initialised weight offset by 1, so it starts at unity but trains
    let gamma = 1.0 + builder.new_weights(format!("{id}_g"), Shape::new(n, 1), InitSettings::Zeroed);
    normed * gamma
}

fn _layer_norm<'a>(builder: &'a ModelBuilder, id: &str, x: ModelNode<'a>) -> ModelNode<'a> {
    const EPS: f32 = 1e-5;
    let n = x.shape().rows();
    assert_eq!(x.shape().cols(), 1, "layer_norm expects a column vector");

    let mean = _broadcast_rows(builder, x.reduce_sum_rows() / n as f32, n);
    let centred = x - mean;

    let var = (centred * centred).reduce_sum_rows() / n as f32;
    let inv_std = _broadcast_rows(builder, (var + EPS).abs_pow(-0.5), n);
    let normed = centred * inv_std;

    let gamma = 1.0 + builder.new_weights(format!("{id}_g"), Shape::new(n, 1), InitSettings::Zeroed);
    let beta = builder.new_weights(format!("{id}_b"), Shape::new(n, 1), InitSettings::Zeroed);
    normed * gamma + beta
}
