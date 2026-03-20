use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, ModelBuilder, ModelNode, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};

const L1: usize = 2560;
const RESIDUAL: usize = 32;
const EXPERT_WIDTH: usize = 32;
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

const BATCH_GLOM: usize = 4;

fn main() {
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

    let mut saves = [
        "l0w", "l0b", "l1w", "l1b", "l2xuw", "l2xub", "l2xdw", "l2xdb", "l2fuw", "l2fub", "l2fdw", "l2fdb", "l3xuw",
        "l3xub", "l3xdw", "l3xdb", "l3fuw", "l3fub", "l3fdw", "l3fdb", "l4xw", "l4fw", "l4xb", "l4fb",
    ]
    .map(SavedFormat::id)
    .to_vec();

    // merge factoriser weights when saving:
    saves[0] = saves[0].clone().transform(|store, mut weights| {
        let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
        weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
    });

    let mut trainer = ValueTrainerBuilder::default()
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .dual_perspective()
        .optimiser(AdamW)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&saves)
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, L1);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // layerstack weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * RESIDUAL);

            // L2 block: shared expert + bucketed expert
            let l2xu = builder.new_affine("l2xu", RESIDUAL, NUM_OUTPUT_BUCKETS * EXPERT_WIDTH * 2);
            let l2xd = builder.new_affine("l2xd", EXPERT_WIDTH, NUM_OUTPUT_BUCKETS * RESIDUAL);
            let l2fu = builder.new_affine("l2fu", RESIDUAL, EXPERT_WIDTH * 2);
            let l2fd = builder.new_affine("l2fd", EXPERT_WIDTH, RESIDUAL);

            // L3 block: shared expert + bucketed expert
            let l3xu = builder.new_affine("l3xu", RESIDUAL, NUM_OUTPUT_BUCKETS * EXPERT_WIDTH * 2);
            let l3xd = builder.new_affine("l3xd", EXPERT_WIDTH, NUM_OUTPUT_BUCKETS * RESIDUAL);
            let l3fu = builder.new_affine("l3fu", RESIDUAL, EXPERT_WIDTH * 2);
            let l3fd = builder.new_affine("l3fd", EXPERT_WIDTH, RESIDUAL);

            // output head
            let l4x = builder.new_affine("l4x", RESIDUAL, NUM_OUTPUT_BUCKETS * HEADS);
            let l4f = builder.new_affine("l4f", RESIDUAL, HEADS);

            // inference
            let stm_subnet = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_subnet.concat(ntm_subnet);

            // L₁-norm penalty on accumulator:
            let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(output_buckets);

            let l1_normed = rms_norm(builder, l1_out, RESIDUAL);

            // L2 shared expert: up → SwiGLU → down
            let l2f_up = l2fu.forward(l1_normed);
            let l2f_out =
                hard_swish(l2f_up.slice_rows(0, EXPERT_WIDTH)) * l2f_up.slice_rows(EXPERT_WIDTH, EXPERT_WIDTH * 2);
            let l2f_out = l2fd.forward(l2f_out);

            // L2 bucketed expert: up → select → SwiGLU → down → select
            let l2x_up = l2xu.forward(l1_normed).select(output_buckets);
            let l2x_out =
                hard_swish(l2x_up.slice_rows(0, EXPERT_WIDTH)) * l2x_up.slice_rows(EXPERT_WIDTH, EXPERT_WIDTH * 2);
            let l2x_out = l2xd.forward(l2x_out).select(output_buckets);

            let l2_out = l2f_out + l2x_out + l1_out;

            let l2_normed = rms_norm(builder, l2_out, RESIDUAL);

            // L3 shared expert: up → SwiGLU → down
            let l3f_up = l3fu.forward(l2_normed);
            let l3f_out =
                hard_swish(l3f_up.slice_rows(0, EXPERT_WIDTH)) * l3f_up.slice_rows(EXPERT_WIDTH, EXPERT_WIDTH * 2);
            let l3f_out = l3fd.forward(l3f_out);

            // L3 bucketed expert: up → select → SwiGLU → down → select
            let l3x_up = l3xu.forward(l2_normed).select(output_buckets);
            let l3x_out =
                hard_swish(l3x_up.slice_rows(0, EXPERT_WIDTH)) * l3x_up.slice_rows(EXPERT_WIDTH, EXPERT_WIDTH * 2);
            let l3x_out = l3xd.forward(l3x_out).select(output_buckets);

            let l3_out = l3f_out + l3x_out + l2_out;

            let l4x_out = l4x.forward(l3_out).select(output_buckets);
            let l4f_out = l4f.forward(l3_out);

            let l4_out = l4x_out + l4f_out;

            if HEADS == 3 {
                // -------- MSE --------
                let loss_mask = builder.new_constant(Shape::new(1, 3), &[1.0, 0.0, 0.0]);
                let draw_mask = builder.new_constant(Shape::new(1, 3), &[0.0, 1.0, 0.0]);
                let win_mask = builder.new_constant(Shape::new(1, 3), &[0.0, 0.0, 1.0]);

                let loss = loss_mask.matmul(l4_out);
                let draw = draw_mask.matmul(l4_out);
                let win = win_mask.matmul(l4_out);

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
                let ce_loss = ones.matmul(l4_out.softmax_crossentropy_loss(targets));

                let loss = mse_loss + 0.1 * ce_loss;

                let loss = loss + 0.005 * l0_out_norm;

                (l4_out, loss)
            } else {
                let loss = l4_out.sigmoid().squared_error(targets);

                let loss = loss + 0.005 * l0_out_norm;

                (l4_out, loss)
            }
        });

    // apply clipping to L1:
    let adamw = AdamWParams { max_weight: CLIP, min_weight: -CLIP, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", adamw);
    trainer.optimiser.set_params_for_weight("l1b", adamw);
    // don't bother clipping the float layers or l0
    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..adamw };
    for name in [
        "l0w", "l0f", "l2xuw", "l2xub", "l2xdw", "l2xdb", "l2fuw", "l2fub", "l2fdw", "l2fdb", "l3xuw", "l3xub",
        "l3xdw", "l3xdb", "l3fuw", "l3fub", "l3fdw", "l3fdb", "l4xw", "l4xb", "l4fw", "l4fb",
    ] {
        trainer.optimiser.set_params_for_weight(name, no_clipping);
    }

    let schedule = TrainingSchedule {
        net_id: "dynode".to_string(),
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

fn exp(x: ModelNode<'_>) -> ModelNode<'_> {
    let sigmoid = x.sigmoid();
    let inv_sigmoid = sigmoid.abs_pow(-1.0);
    let e_minus_x = inv_sigmoid - 1.0;
    e_minus_x.abs_pow(-1.0)
}

fn rms_norm<'a>(builder: &'a ModelBuilder, x: ModelNode<'a>, dim: usize) -> ModelNode<'a> {
    let mean_coeff = builder.new_constant(Shape::new(1, dim), &vec![1.0 / dim as f32; dim]);
    let mean_sq = mean_coeff.matmul(x * x);
    let inv_rms = (mean_sq + 1e-6).abs_pow(-0.5);
    let ones_col = builder.new_constant(Shape::new(dim, 1), &vec![1.0; dim]);
    x * ones_col.matmul(inv_rms)
}

fn hard_swish<'a>(x: ModelNode<'a>) -> ModelNode<'a> {
    let gate = (x * 1. / 6.0 + 0.5).crelu();
    x * gate
}
