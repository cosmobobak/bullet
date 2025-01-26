use bullet_lib::{
    default::{inputs::ChessBucketsMirroredFactorised, outputs::MaterialCount},
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        Activation, ExecutionContext, Graph, NetworkBuilder, Node, Shape,
    },
    optimiser::Optimiser,
    trainer::{
        default::{inputs, loader, outputs, Trainer},
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    NetworkTrainer,
};

const HL: usize = 2048;
const L2: usize = 16;
const L3: usize = 32;

const FINE_TUNING: bool = true;

type Input = ChessBucketsMirroredFactorised;
type Output = MaterialCount<8>;

fn main() {
    #[rustfmt::skip]
    let inputs = Input::new([
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
         8,  9, 10, 11,
        12, 12, 13, 13,
        12, 12, 13, 13,
        14, 14, 15, 15,
        14, 14, 15, 15,
    ]);
    let output_buckets = Output::default();

    let num_inputs = <Input as inputs::SparseInputType>::num_inputs(&inputs);
    let num_buckets = <Output as outputs::OutputBuckets<_>>::BUCKETS;

    // let (mut graph, output_node) = build_network(inputs.size(), HL, 8);
    let (graph, output_node) = build_network(num_inputs, num_buckets, HL);

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        output_buckets,
        vec![
            SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3b", QuantTarget::Float, Layout::Normal),
        ],
        false,
    );

    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..AdamWParams::default() };

    trainer.optimiser_mut().set_params_for_weight("l2w", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l2b", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3w", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3b", no_clipping);

    trainer.load_from_checkpoint("checkpoints/temperance-200");

    let initial_lr;
    let final_lr;
    let sbs;
    if FINE_TUNING {
        initial_lr = 0.0005;
        final_lr = 0.0005 * 0.3 * 0.3 * 0.3;
        sbs = 200;
    } else {
        initial_lr = 0.001;
        final_lr = 0.001 * 0.3 * 0.3 * 0.3;
        sbs = 800;
    }

    let schedule = TrainingSchedule {
        net_id: "perseverance".into(),
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        eval_scale: 400.0,
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: sbs },
            warmup_batches: 800,
        },
        save_rate: 200,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/dataset.bin"]);

    trainer.run(&schedule, &settings, &data_loader);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}

fn build_network(num_inputs: usize, output_buckets: usize, hl: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_input("stm", Shape::new(num_inputs, 1));
    let nstm = builder.new_input("nstm", Shape::new(num_inputs, 1));
    let targets = builder.new_input("targets", Shape::new(1, 1));
    let buckets = builder.new_input("buckets", Shape::new(output_buckets, 1));

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, hl);
    let l1 = builder.new_affine("l1", hl, output_buckets * L2);
    let l2 = builder.new_affine("l2", L2, output_buckets * L3);
    let l3 = builder.new_affine("l3", L3, output_buckets);

    // inference
    let out = l0.forward_sparse_dual_with_activation(stm, nstm, Activation::CReLU);
    let out = out.pairwise_mul_post_affine_dual();
    let out = l1.forward(out).select(buckets).activate(Activation::SCReLU);
    let out = l2.forward(out).select(buckets).activate(Activation::SCReLU);
    let out = l3.forward(out).select(buckets);

    let pred = out.activate(Activation::Sigmoid);
    pred.mse(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
