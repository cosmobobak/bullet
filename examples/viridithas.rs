use bullet_lib::{
    game::{inputs::SparseInputType as _, outputs::MaterialCount},
    nn::{
        ModelNode, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};

use crate::threat_inputs::ThreatInputs;

mod attacks;
mod indices;
mod offsets;
mod threat_inputs;
mod threats;

const L1: usize = 1024;
const L2: usize = 32;
const L3: usize = 32;
const HEADS: usize = 1;

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

fn main() {
    let inputs = ThreatInputs::new(BUCKET_LAYOUT);

    // hyperparams to fiddle with
    let dataset_path = "data/all.vf";
    let initial_lr = 0.001;
    let superbatches = 800;
    let lr_scheduler = lr::Warmup {
        inner: lr::CosineDecayLR {
            initial_lr,
            final_lr: initial_lr * f32::powi(0.3, 5),
            final_superbatch: superbatches,
        },
        warmup_batches: 1200,
    };
    let wdl_scheduler = wdl::LinearWDL { start: 0.4, end: 1.0 };

    let saves = ["l0w", "l0b", "l1w", "l1b", "l2xw", "l2fw", "l2xb", "l2fb", "l3xw", "l3fw", "l3xb", "l3fb"]
        .map(SavedFormat::id);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .optimiser(AdamW)
        .save_format(&saves)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1);
            l0.init_with_effective_input_size(20000);

            // layerstack weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * L2);
            let l2x = builder.new_affine("l2x", L2, NUM_OUTPUT_BUCKETS * L3 * 2);
            let l2f = builder.new_affine("l2f", L2, L3 * 2);
            let l3x = builder.new_affine("l3x", L3, NUM_OUTPUT_BUCKETS * HEADS);
            let l3f = builder.new_affine("l3f", L3, HEADS);

            // inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let l0_out = stm_subnet.concat(ntm_subnet);

            // L₁-norm penalty on accumulator (mean, since values are non-negative):
            let mean_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = mean_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(buckets);
            let l1_out = hard_swish(l1_out);

            let l2x_out = l2x.forward(l1_out).select(buckets);
            let l2f_out = l2f.forward(l1_out);
            let l2_out = l2x_out + l2f_out;
            // SwiGLU: l2_out = W₁x · Swish(W₂x)
            let l2_out = hard_swish(l2_out.slice_rows(0, L3)) * l2_out.slice_rows(L3, L3 * 2);

            // skip connexion from l1-out to l2-out:
            let l2_out = l2_out + l1_out;

            let l3x_out = l3x.forward(l2_out).select(buckets);
            let l3f_out = l3f.forward(l2_out);

            let l3_out = l3x_out + l3f_out;

            if HEADS == 3 {
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

                let loss = loss + 0.005 * l0_out_norm;

                (l3_out, loss)
            } else {
                let loss = l3_out.sigmoid().squared_error(targets);

                let loss = loss + 0.005 * l0_out_norm;

                (l3_out, loss)
            }
        });

    let default_optimiser_params =
        AdamWParams { beta1: 0.99, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, ..Default::default() };
    let l0w_optimiser_params = AdamWParams { min_weight: -0.99, max_weight: 0.99, ..default_optimiser_params };
    let l1w_clip = 0.99 * 255.0 * 255.0 / (256.0 * 256.0);
    let l1w_optimiser_params = AdamWParams { min_weight: -l1w_clip, max_weight: l1w_clip, ..default_optimiser_params };
    trainer.optimiser.set_params(default_optimiser_params);
    trainer.optimiser.set_params_for_weight("l0w", l0w_optimiser_params);
    trainer.optimiser.set_params_for_weight("l1w", l1w_optimiser_params);
    // don't bother clipping the float layers
    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..default_optimiser_params };
    for name in ["l2xw", "l2xb", "l2fw", "l2fb", "l3xw", "l3xb", "l3fw", "l3fb"] {
        trainer.optimiser.set_params_for_weight(name, no_clipping);
    }

    let schedule = TrainingSchedule {
        net_id: "augury".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384 * BATCH_GLOM,
            batches_per_superbatch: 6104 / BATCH_GLOM,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler,
        lr_scheduler,
        save_rate: 10000,
    };

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

    trainer.run(&schedule, &settings, &dataloader);
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
