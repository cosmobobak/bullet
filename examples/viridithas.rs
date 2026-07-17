use bullet_lib::{
    game::{inputs::SparseInputType as _, outputs::MaterialCount},
    nn::{
        InitSettings, ModelBuilder, ModelNode, Shape,
        optimiser::{Ranger, RangerParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};

use crate::pawn_pawn_inputs::PawnPawnInputs;

mod attacks;
mod indices;
mod offsets;
mod pawn_pawn_inputs;
mod threat_inputs;
mod threats;

const NET_ID: &str = "magnetar";

const L1: usize = 1024;
const D: usize = 32;
const HIDDEN: usize = D;
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

const BATCH_GLOM: usize = 4;

// values verbatim from a pawnocchio schedule
const SUPERBATCHES_STAGE0: usize = 100;
const SUPERBATCHES_STAGE1: usize = 800;
const SUPERBATCHES_STAGE2: usize = 200;

fn main() {
    let inputs = PawnPawnInputs::new(BUCKET_LAYOUT, pawn_pawn_inputs::three_file_band_mask());

    // hyperparams to fiddle with
    let dataset_path = "data/all.vf";

    #[rustfmt::skip]
    let saves = [
        "l0w", "l0b", "l1w", "l1b",
        "l1n_g",
        "l2up_xw", "l2up_fw", "l2up_xb", "l2up_fb",
        "l2down_xw", "l2down_fw", "l2down_xb", "l2down_fb",
        "l2n_g",
        "l3up_xw", "l3up_fw", "l3up_xb", "l3up_fb",
        "l3down_xw", "l3down_fw", "l3down_xb", "l3down_fb",
        "l4xw", "l4fw", "l4xb", "l4fb",
    ]
    .map(SavedFormat::id);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .optimiser(Ranger)
        .full_output()
        .save_format(&saves)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1);
            l0.init_with_effective_input_size(20000);

            // layerstack weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * D);
            // block A
            let l2up_x = builder.new_affine("l2up_x", D, NUM_OUTPUT_BUCKETS * HIDDEN * 2);
            let l2up_f = builder.new_affine("l2up_f", D, HIDDEN * 2);
            let l2down_x = builder.new_affine("l2down_x", HIDDEN, NUM_OUTPUT_BUCKETS * D);
            let l2down_f = builder.new_affine("l2down_f", HIDDEN, D);
            // block B
            let l3up_x = builder.new_affine("l3up_x", D, NUM_OUTPUT_BUCKETS * HIDDEN * 2);
            let l3up_f = builder.new_affine("l3up_f", D, HIDDEN * 2);
            let l3down_x = builder.new_affine("l3down_x", HIDDEN, NUM_OUTPUT_BUCKETS * D);
            let l3down_f = builder.new_affine("l3down_f", HIDDEN, D);
            // head
            let l4x = builder.new_affine("l4x", D, NUM_OUTPUT_BUCKETS * HEADS);
            let l4f = builder.new_affine("l4f", D, HEADS);
            // auxiliary WDL-classification head, training-only (not saved)
            // let l3wdl_x = builder.new_affine("l3wdl_x", D, NUM_OUTPUT_BUCKETS * 3);
            // let l3wdl_f = builder.new_affine("l3wdl_f", D, 3);

            // inference
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_subnet = ft(stm, 0, L1 / 2) * ft(stm, L1 / 2, L1);
            let ntm_subnet = ft(ntm, 0, L1 / 2) * ft(ntm, L1 / 2, L1);
            let l0_out = stm_subnet.concat(ntm_subnet);

            // L₁-norm penalty on accumulator (mean, since values are non-negative):
            let mean_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = mean_l1_vec.matmul(l0_out);

            // note: deliberately not activating l1_out.
            let l1_out = l1.forward(l0_out).select(buckets);

            // BLOCK A (pre-norm FFN_SwiGLU(x))
            let h = rms_norm(builder, "l1n", l1_out);
            let p = l2up_x.forward(h).select(buckets) + l2up_f.forward(h);
            let gate = hard_swish(p.slice_rows(0, HIDDEN));
            let id = p.slice_rows(HIDDEN, HIDDEN * 2);
            let g = gate * id;
            let l2_out = l2down_x.forward(g).select(buckets) + l2down_f.forward(g) + l1_out;

            // BLOCK B
            let h = rms_norm(builder, "l2n", l2_out);
            let p = l3up_x.forward(h).select(buckets) + l3up_f.forward(h);
            let gate = hard_swish(p.slice_rows(0, HIDDEN));
            let id = p.slice_rows(HIDDEN, HIDDEN * 2);
            let g = gate * id;
            let l3_out = l3down_x.forward(g).select(buckets) + l3down_f.forward(g) + l2_out;

            // read output from stream
            let out = l4x.forward(l3_out).select(buckets) + l4f.forward(l3_out);

            if HEADS == 3 {
                // -------- MSE --------
                let loss_mask = builder.new_constant(Shape::new(1, 3), &[1.0, 0.0, 0.0]);
                let draw_mask = builder.new_constant(Shape::new(1, 3), &[0.0, 1.0, 0.0]);
                let win_mask = builder.new_constant(Shape::new(1, 3), &[0.0, 0.0, 1.0]);

                let loss = loss_mask.matmul(out);
                let draw = draw_mask.matmul(out);
                let win = win_mask.matmul(out);

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
                let ce_loss = ones.matmul(out.softmax_crossentropy_loss(targets));

                let loss = mse_loss + 0.1 * ce_loss;

                let loss = loss + 0.005 * l0_out_norm;

                (out, loss)
            } else {
                // targets: row 0 is the WDL-blended value, rows 1..4 one-hot game result
                let target_value = targets.slice_rows(0, 1);
                // let target_wdl = targets.slice_rows(1, 4);

                let value_loss = out.sigmoid().squared_error(target_value);

                // let wdl_logits = l3wdl_x.forward(l3_out).select(buckets) + l3wdl_f.forward(l3_out);
                // let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
                // let wdl_loss = ones.matmul(wdl_logits.softmax_crossentropy_loss(target_wdl));
                // let wdl_logit_norm = ones.matmul(wdl_logits * wdl_logits);

                let loss = value_loss + 0.005 * l0_out_norm; // + WDL_CE_ALPHA * wdl_loss + WDL_Z_BETA * wdl_logit_norm;

                (out, loss)
            }
        });

    let default_optimiser_params =
        RangerParams { beta1: 0.99, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, ..Default::default() };
    let l0w_optimiser_params = RangerParams { min_weight: -0.99, max_weight: 0.99, ..default_optimiser_params };
    let l1w_clip = 0.99 * 255.0 * 255.0 / (256.0 * 256.0);
    let l1w_optimiser_params = RangerParams { min_weight: -l1w_clip, max_weight: l1w_clip, ..default_optimiser_params };
    trainer.optimiser.set_params(default_optimiser_params);
    trainer.optimiser.set_params_for_weight("l0w", l0w_optimiser_params);
    trainer.optimiser.set_params_for_weight("l1w", l1w_optimiser_params);
    // don't bother clipping the float layers
    let no_clipping = RangerParams { min_weight: -128.0, max_weight: 128.0, ..default_optimiser_params };
    #[rustfmt::skip]
    let noclip_names = [
        "l1n_g", "l2n_g",
        "l2up_xw", "l2up_xb", "l2up_fw", "l2up_fb",
        "l2down_xw", "l2down_xb", "l2down_fw", "l2down_fb",
        "l3up_xw", "l3up_xb", "l3up_fw", "l3up_fb",
        "l3down_xw", "l3down_xb", "l3down_fw", "l3down_fb",
        "l4xw", "l4xb", "l4fw", "l4fb",
        // "l3wdl_xw", "l3wdl_xb", "l3wdl_fw", "l3wdl_fb",
    ];
    for name in noclip_names {
        trainer.optimiser.set_params_for_weight(name, no_clipping);
    }

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader = bullet_lib::value::loader::ViriBinpackLoader::new(
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

    const WARMUP_SBS: usize = SUPERBATCHES_STAGE0 / 2;
    const COOLDOWN_SBS: usize = SUPERBATCHES_STAGE0 - WARMUP_SBS;
    trainer.run(
        &stage_schedule(
            format!("{}-s0", NET_ID),
            SUPERBATCHES_STAGE0,
            wdl::ConstantWDL { value: 0.2 },
            lr::Sequence {
                first: lr::LinearDecayLR { initial_lr: 1e-4, final_lr: 5e-3, final_superbatch: WARMUP_SBS },
                second: lr::LinearDecayLR { initial_lr: 5e-3, final_lr: 1e-4, final_superbatch: COOLDOWN_SBS },
                first_scheduler_final_superbatch: WARMUP_SBS,
            },
        ),
        &settings,
        &dataloader,
    );

    trainer.run(
        &stage_schedule(
            format!("{}-s1", NET_ID),
            SUPERBATCHES_STAGE1,
            wdl::LinearWDL { start: 0.2, end: 0.5 },
            lr::LinearDecayLR { initial_lr: 1e-3, final_lr: 1e-6, final_superbatch: SUPERBATCHES_STAGE1 },
        ),
        &settings,
        &dataloader,
    );

    trainer.run(
        &stage_schedule(
            format!("{}-s2", NET_ID),
            SUPERBATCHES_STAGE2,
            wdl::ConstantWDL { value: 1.0 },
            lr::LinearDecayLR { initial_lr: 1e-5, final_lr: 1e-7, final_superbatch: SUPERBATCHES_STAGE2 },
        ),
        &settings,
        &dataloader,
    );
}

fn stage_schedule<LR: lr::LrScheduler, WDL: wdl::WdlScheduler>(
    net_id: String,
    end_superbatch: usize,
    wdl_scheduler: WDL,
    lr_scheduler: LR,
) -> TrainingSchedule<LR, WDL> {
    TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384 * BATCH_GLOM,
            batches_per_superbatch: 6104 / BATCH_GLOM,
            start_superbatch: 1,
            end_superbatch,
        },
        wdl_scheduler,
        lr_scheduler,
        // we could set this lower for SWA,
        // but tests indicate it doesn’t really help.
        // https://viri.dev/test/694/
        //   LLR −2.95 (−2.94 LO +2.94 HI BND for +0.00 LO +3.00 HI ELO)
        //  PERF −0.77 ± 1.74 ELO = −2.51 LO +0.96 HI ELO
        //  CONF 8+0.08 SEC 1 THREAD 16 MB CACHE
        // GAMES 39056 = 9744W 19481D 9831L = 24.9%W 49.9%D 25.2%L
        // PENTA 85⁺² 4586⁺¹ 10086⁰ 4699⁻¹ 72⁻²
        save_rate: 10000,
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

fn broadcast_rows<'a>(builder: &'a ModelBuilder, scalar: ModelNode<'a>, rows: usize) -> ModelNode<'a> {
    let ones = builder.new_constant(Shape::new(rows, 1), &vec![1.0; rows]);
    ones.matmul(scalar)
}

fn rms_norm<'a>(builder: &'a ModelBuilder, id: &str, x: ModelNode<'a>) -> ModelNode<'a> {
    const EPS: f32 = 1e-5;
    let n = x.shape().rows();
    assert_eq!(x.shape().cols(), 1, "rms_norm expects a column vector");

    // mean of squares over the feature dimension (rows), broadcast back to (n, 1)
    let mean_sq = (x * x).reduce_sum_rows() / n as f32;
    let inv_rms = broadcast_rows(builder, (mean_sq + EPS).abs_pow(-0.5), n);
    let normed = x * inv_rms;

    // γ as a 0-initialised weight offset by 1, so it starts at unity but trains
    let gamma = 1.0 + builder.new_weights(format!("{id}_g"), Shape::new(n, 1), InitSettings::Zeroed);
    normed * gamma
}

fn _layer_norm<'a>(builder: &'a ModelBuilder, id: &str, x: ModelNode<'a>) -> ModelNode<'a> {
    const EPS: f32 = 1e-5;
    let n = x.shape().rows();
    assert_eq!(x.shape().cols(), 1, "layer_norm expects a column vector");

    let mean = broadcast_rows(builder, x.reduce_sum_rows() / n as f32, n);
    let centred = x - mean;

    let var = (centred * centred).reduce_sum_rows() / n as f32;
    let inv_std = broadcast_rows(builder, (var + EPS).abs_pow(-0.5), n);
    let normed = centred * inv_std;

    let gamma = 1.0 + builder.new_weights(format!("{id}_g"), Shape::new(n, 1), InitSettings::Zeroed);
    let beta = builder.new_weights(format!("{id}_b"), Shape::new(n, 1), InitSettings::Zeroed);
    normed * gamma + beta
}
