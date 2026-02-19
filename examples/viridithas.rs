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

use acyclib::{
    dag::NodeId,
    device::function::{self, DeviceFunction},
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        },
    },
};
use bullet_cuda_backend::{
    CudaDevice,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

#[derive(Debug)]
struct BCELogitsLoss {
    logits: AnnotatedNode,
    target: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for BCELogitsLoss {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.logits, self.target]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_same_batching(ir, &[&self.logits, &self.target])?;
        util::check_dense_eq(ir, &self.logits, true)?;
        util::check_dense_eq(ir, &self.target, true)?;

        if self.logits.shape == self.target.shape {
            Ok(self.logits.shape)
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.logits.shape,
                self.target.shape,
            ])))
        }
    }
}

impl GraphIROperationCompilable<CudaMarker> for BCELogitsLoss {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let logits = graph.get_ref(self.logits.idx, GraphNodeIdTy::Values);
        let target = graph.get_ref(self.target.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: logits.clone(), output: output.clone() });

        let code = "
            __device__ float ln_sigmoid(float x) {{
                return x >= 0.0 ? -logf(1.0 + expf(-x)) : x - logf(1.0 + expf(x));
            }}

            extern \"C\" __global__ void kernel(const int size, const float* logits, const float* target, float* output)
            {{
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < size)
                {{
                    const float inp = logits[tid];
                    const float tar = target[tid];

                    output[tid] = tar == 0.0f
                        ? -ln_sigmoid(-inp)
                        : tar == 1.0f
                            ? -ln_sigmoid(inp)
                            : tar * (logf(tar) - ln_sigmoid(inp)) + (1.0 - tar) * (logf(1.0 - tar) - ln_sigmoid(-inp));
                }}
            }}";

        let threads = Expr::Const(512);
        let size = Expr::Var * logits.single_size() as i32;
        let blocks = (size.clone() + threads.clone() - 1) / threads.clone();
        let grid_dim = [blocks, Expr::Const(1), Expr::Const(1)];
        let block_dim = [threads, Expr::Const(1), Expr::Const(1)];

        let layout = None;
        let batched = logits.batch_size().is_some();
        let shape = logits.shape();
        let inputs = vec![
            KernelInput::Size(size),
            KernelInput::Slice { slice: logits, layout, mutable: false, batched, shape },
            KernelInput::Slice { slice: target, layout, mutable: false, batched, shape },
            KernelInput::Slice { slice: output, layout, mutable: true, batched, shape },
        ];

        let args = KernelArgs { grid_dim, block_dim, shared_mem_bytes: Expr::Const(0), inputs };

        let kernel = unsafe { Kernel::new("BCELogitsLossFwd".to_string(), code.into(), args) };

        func.push(kernel.unwrap());

        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let logits = graph.get_ref(self.logits.idx, GraphNodeIdTy::Values);
        let target = graph.get_ref(self.target.idx, GraphNodeIdTy::Values);
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let mut func = DeviceFunction::default();

        if let Some(input_grad) = graph.maybe_get_ref(self.logits.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: output_grad.clone(), output: input_grad.clone() });

            let code = "
                extern \"C\" __global__ void kernel(const int size, const float* logits, const float* target, const float* ogrd, float* igrd)
                {{
                    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                    if (tid < size)
                    {{
                        const float sig = 1.0f / (1.0f + expf(-logits[tid]));
                        const float tar = target[tid];
                        igrd[tid] += ogrd[tid] * (-tar * (1.0 - sig) + (1.0 - tar) * sig);
                    }}
                }}";

            let threads = Expr::Const(512);
            let size = Expr::Var * logits.single_size() as i32;
            let blocks = (size.clone() + threads.clone() - 1) / threads.clone();
            let grid_dim = [blocks, Expr::Const(1), Expr::Const(1)];
            let block_dim = [threads, Expr::Const(1), Expr::Const(1)];

            let layout = None;
            let batched = logits.batch_size().is_some();
            let shape = logits.shape();
            let inputs = vec![
                KernelInput::Size(size),
                KernelInput::Slice { slice: logits, layout, mutable: false, batched, shape },
                KernelInput::Slice { slice: target, layout, mutable: false, batched, shape },
                KernelInput::Slice { slice: output_grad, layout, mutable: false, batched, shape },
                KernelInput::Slice { slice: input_grad, layout, mutable: true, batched, shape },
            ];

            let args = KernelArgs { grid_dim, block_dim, shared_mem_bytes: Expr::Const(0), inputs };

            let kernel = unsafe { Kernel::new("BCELogitsLossFwd".to_string(), code.into(), args) };

            func.push(kernel.unwrap());
        }

        if graph.maybe_get_ref(self.target.idx, GraphNodeIdTy::Gradients).is_some() {
            panic!("Unsupported!");
        }

        func
    }
}

const L1: usize = 2560;
const L2: usize = 16;
const L3: usize = 32;
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
            final_lr: initial_lr * f32::powi(0.3, 3),
            final_superbatch: superbatches,
        },
        warmup_batches: 1600,
    };
    let wdl_scheduler = wdl::LinearWDL { start: 0.4, end: 1.0 };
    // let wdl_scheduler = wdl::ConstantWDL { value: 1.0 };

    let mut saves = ["l0w", "l0b", "l1w", "l1b", "l2xw", "l2fw", "l2xb", "l2fb", "l3xw", "l3fw", "l3xb", "l3fb"]
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
            l0.weights =
                (l0.weights + expanded_factoriser).clip_pass_through_grad(-CLIP, CLIP).faux_quantise(255.0, true);
            l0.bias = l0.bias.faux_quantise(255.0, true);

            // layerstack weights
            let mut l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * L2);
            l1.weights = l1.weights.faux_quantise(64.0, true);
            let l2x = builder.new_affine("l2x", L2, NUM_OUTPUT_BUCKETS * L3 * 2);
            let l2f = builder.new_affine("l2f", L2, L3 * 2);
            let l3x = builder.new_affine("l3x", L3, NUM_OUTPUT_BUCKETS * HEADS);
            let l3f = builder.new_affine("l3f", L3, HEADS);

            // inference
            let stm_subnet = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_subnet.concat(ntm_subnet).faux_quantise(127.0, false);

            // L₁-norm penalty on accumulator:
            let ones_l1_vec = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let l1_out = hard_swish(l1_out);

            let l2x_out = l2x.forward(l1_out).select(output_buckets);
            let l2f_out = l2f.forward(l1_out);
            let l2_out = l2x_out + l2f_out;
            // SwiGLU: l2_out = W₁x · Swish(W₂x)
            let l2_swish = hard_swish(l2_out.slice_rows(0, L3));
            let l2_ident = l2_out.slice_rows(L3, L3 * 2);
            let l2_out = l2_swish * l2_ident;

            let l3x_out = l3x.forward(l2_out).select(output_buckets);
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
                // let loss =
                //     builder.apply(BCELogitsLoss { logits: l3_out.annotated_node(), target: targets.annotated_node() });
                let loss = l3_out.sigmoid().squared_error(targets);

                let loss = loss + 0.005 * l0_out_norm;

                (l3_out, loss)
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

    let schedule = TrainingSchedule {
        net_id: "click".to_string(),
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

fn hard_swish(x: GraphBuilderNode<'_, CudaMarker>) -> GraphBuilderNode<'_, CudaMarker> {
    let relu6 = ((x + 3.0) / 6.0).crelu() * 6.0;
    x * relu6 / 6.0
}
