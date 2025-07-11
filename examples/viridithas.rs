use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        ExecutionContext, Graph, InitSettings, NetworkBuilder, Node, Shape,
    },
    trainer::{
        default::{inputs, loader, outputs, Trainer},
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
};
use bulletformat::ChessBoard;

const HL: usize = 2048;
const L2: usize = 16;
const L3: usize = 32;

const FINE_TUNING: bool = false;

const CLIP: f32 = 0.99 * 2.0;

type Input = ChessBucketsMirrored;
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

    let num_inputs = inputs::SparseInputType::num_inputs(&inputs);
    let max_active = inputs::SparseInputType::max_active(&inputs);
    let num_buckets = <Output as outputs::OutputBuckets<ChessBoard>>::BUCKETS;

    let (graph, loss_node, mean_node, var_node) = build_network(num_inputs, max_active, num_buckets, HL);

    let adamw = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -CLIP, max_weight: CLIP };

    let mut saves = [
        "l0w", "l0b", // input layer weights
        "l1xw", "l1fw", "l1xb", "l1fb", // l1 weights
        "l2xw", "l2fw", "l2xb", "l2fb", // l2 weights
        "l3xw", "l3fw", "l3xb", "l3fb", // l3 weights
        "e3xw", "e3fw", "e3xb", "e3fb", // variance head weights
    ]
    .map(SavedFormat::id)
    .to_vec();

    // merge factoriser weights when saving:
    saves[0] = saves[0].clone().add_transform(|builder, _, mut weights| {
        let factoriser = builder.get_weights("l0f").get_dense_vals().unwrap();
        let expanded = factoriser.repeat(weights.len() / factoriser.len());

        for (i, j) in weights.iter_mut().zip(expanded.iter()) {
            *i += *j;
        }

        weights
    });

    let mut trainer =
        Trainer::<AdamWOptimiser, _, _>::new(graph, loss_node, adamw, inputs, output_buckets, saves, false);

    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..adamw };

    // l2
    trainer.optimiser_mut().set_params_for_weight("l2xw", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l2xb", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l2fw", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l2fb", no_clipping);

    // l3
    trainer.optimiser_mut().set_params_for_weight("l3xw", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3xb", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3fw", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3fb", no_clipping);

    // var head
    trainer.optimiser_mut().set_params_for_weight("e3xw", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("e3xb", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("e3fw", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("e3fb", no_clipping);

    let initial_lr;
    let final_lr;
    let sbs;
    if FINE_TUNING {
        initial_lr = 0.0005;
        final_lr = 0.0005 * f32::powi(0.3, 6);
        sbs = 200;
    } else {
        initial_lr = 0.001;
        final_lr = 0.001 * f32::powi(0.3, 6);
        sbs = 800;
    }

    let schedule = TrainingSchedule {
        net_id: "metaheuristic".to_string(),
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        eval_scale: 400.0,
        wdl_scheduler: wdl::LinearWDL { start: 0.4, end: 0.65 },
        lr_scheduler: lr::Warmup {
            inner: lr::LinearDecayLR { initial_lr, final_lr, final_superbatch: sbs },
            warmup_batches: 1600,
        },
        save_rate: 200,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/dataset.bin"]);

    // trainer.load_from_checkpoint("checkpoints/delenda-800");
    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval_raw_output_for_node(fen, mean_node);
        let var = trainer.eval_raw_output_for_node(fen, var_node);
        println!("FEN : {fen}");
        println!("EVAL: {}", 400.0 * eval[0]);
        println!("VAR : {}", 400.0 * var[0]);
    }
}

fn build_network(num_inputs: usize, max_active: usize, output_buckets: usize, hl: usize) -> (Graph, Node, Node, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(output_buckets, 1), 1);

    // input layer factoriser
    let l0f = builder.new_weights(
        "l0f",
        Shape::new(hl, 768),
        InitSettings::Normal { mean: 0.0, stdev: (2.0 / 32_f32).sqrt() },
    );
    let expanded_factoriser = l0f.repeat(16);

    // input layer weights
    let mut l0 = builder.new_affine("l0", num_inputs, hl);
    l0.init_with_effective_input_size(32);
    l0.weights = l0.weights + expanded_factoriser;

    // trainable weights
    // let l0 = builder.new_affine("l0", num_inputs, hl);
    let l1x = builder.new_affine("l1x", hl, output_buckets * L2);
    let l1f = builder.new_affine("l1f", hl, L2);
    let l2x = builder.new_affine("l2x", L2, output_buckets * L3);
    let l2f = builder.new_affine("l2f", L2, L3);
    let l3x = builder.new_affine("l3x", L3, output_buckets);
    let l3f = builder.new_affine("l3f", L3, 1);
    let e3x = builder.new_affine("e3x", L3, output_buckets);
    let e3f = builder.new_affine("e3f", L3, 1);

    // inference
    let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
    let ntm_subnet = l0.forward(nstm).crelu().pairwise_mul();
    let accumulator = stm_subnet.concat(ntm_subnet);

    let l1x_out = l1x.forward(accumulator).select(buckets);
    let l1f_out = l1f.forward(accumulator);
    let l1_out = (l1x_out + l1f_out).screlu();

    let l2x_out = l2x.forward(l1_out).select(buckets);
    let l2f_out = l2f.forward(l1_out);
    let l2_out = (l2x_out + l2f_out).screlu();

    let l3x_out = l3x.forward(l2_out).select(buckets);
    let l3f_out = l3f.forward(l2_out);
    let l3_out = l3x_out + l3f_out;

    // difference between the output and the target
    let mean_loss = l3_out.sigmoid().squared_error(targets);

    let mean_node = l3_out.node();

    // (builder.build(ExecutionContext::default()), value_node)

    // variance head
    let e3x_out = e3x.forward(l2_out).select(buckets);
    let e3f_out = e3f.forward(l2_out);
    let error_out = e3x_out + e3f_out;
    let var_node = error_out.node();

    // mean loss is at most 1.0 (if net predicts x and actual is 1 - x)
    // with a sane net it's really at most 0.25 (as guessing 0.5 guarantees this)
    // as such, we scale and clip the loss to make it nice and learnable.
    // losses of 0.25 or more become 1.0, and e.g. a loss of 0.05 becomes 0.2.
    // this means that the pre-sigmoid units of the variance head are not raw
    // 400-ths of a centipawn like the mean head. not sure how to work this backwards.
    let scaled_loss = (mean_loss.copy_stop_grad() * 4.0).crelu();
    let variance_loss = error_out.sigmoid().squared_error(scaled_loss);

    // recombine outputs
    let loss_sum = mean_loss + 0.1 * variance_loss;
    let loss_node = loss_sum.node();

    (builder.build(ExecutionContext::default()), loss_node, mean_node, var_node)
}
