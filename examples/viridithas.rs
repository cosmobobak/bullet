use acyclib::graph::builder::GraphBuilderNode;
use bullet_cuda_backend::CudaMarker;
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

const L1: usize = 2560;
const L2: usize = 16;
const L3: usize = 32;
const L4: usize = 32;
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
        warmup_batches: 1600,
    };
    let wdl_scheduler = wdl::LinearWDL { start: 0.4, end: 1.0 };
    // let wdl_scheduler = wdl::ConstantWDL { value: 1.0 };

    let mut saves = [
        "l0w", "l0b", "l1w", "l1b", "l2xw", "l2fw", "l2xb", "l2fb", "l3xw", "l3fw", "l3xb", "l3fb", "l4xw", "l4fw",
        "l4xb", "l4fb",
    ]
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
        // .full_output()
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .dual_perspective()
        .optimiser(AdamW)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&saves)
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), targets| {
            // builder.dump_graphviz("viz.txt");
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, L1);
            l0.init_with_effective_input_size(32);
            l0.weights = (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP);

            // layerstack weights
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * L2);
            let l2x = builder.new_affine("l2x", L2, NUM_OUTPUT_BUCKETS * L3 * 2);
            let l2f = builder.new_affine("l2f", L2, L3 * 2);
            let l3x = builder.new_affine("l3x", L3, NUM_OUTPUT_BUCKETS * L4 * 2);
            let l3f = builder.new_affine("l3f", L3, L4 * 2);
            let l4x = builder.new_affine("l4x", L4, NUM_OUTPUT_BUCKETS * HEADS);
            let l4f = builder.new_affine("l4f", L4, HEADS);

            // inference
            let stm_subnet = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_subnet.concat(ntm_subnet);

            // L₁-norm penalty on accumulator:
            let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let l1_out = hswish(l1_out);

            let l2x_out = l2x.forward(l1_out).select(output_buckets);
            let l2f_out = l2f.forward(l1_out);
            let l2_out = l2x_out + l2f_out;
            // SwiGLU: l2_out = W₁x · Swish(W₂x)
            let l2_out = hswish(l2_out.slice_rows(0, L3)) * l2_out.slice_rows(L3, L3 * 2);

            let l3x_out = l3x.forward(l2_out).select(output_buckets);
            let l3f_out = l3f.forward(l2_out);
            let l3_out = l3x_out + l3f_out;
            // SwiGLU: l3_out = W₁x · Swish(W₂x)
            let l3_out = hswish(l3_out.slice_rows(0, L4)) * l3_out.slice_rows(L4, L4 * 2);

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
    trainer.optimiser.set_params_for_weight("l4xw", no_clipping);
    trainer.optimiser.set_params_for_weight("l4xb", no_clipping);
    trainer.optimiser.set_params_for_weight("l4fw", no_clipping);
    trainer.optimiser.set_params_for_weight("l4fb", no_clipping);

    let schedule = TrainingSchedule {
        net_id: "vertigo".to_string(),
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

    // trainer.load_from_checkpoint("checkpoints/flounce-800");

    trainer.run(&schedule, &settings, &dataloader);
}

fn maximum<'a>(
    x: GraphBuilderNode<'a, CudaMarker>,
    y: GraphBuilderNode<'a, CudaMarker>,
) -> GraphBuilderNode<'a, CudaMarker> {
    (x - y).relu() + y
}

fn exp(x: GraphBuilderNode<'_, CudaMarker>) -> GraphBuilderNode<'_, CudaMarker> {
    let sigmoid = x.sigmoid();
    let inv_sigmoid = sigmoid.abs_pow(-1.0);
    let e_minus_x = inv_sigmoid - 1.0;
    e_minus_x.abs_pow(-1.0)
}

fn hswish(x: GraphBuilderNode<'_, CudaMarker>) -> GraphBuilderNode<'_, CudaMarker> {
    let relu6 = ((x + 3.0) / 6.0).crelu() * 6.0;
    x * relu6 / 6.0
}
