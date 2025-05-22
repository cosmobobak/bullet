use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        ExecutionContext, Graph, NetworkBuilder, Node, Shape,
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

const FINE_TUNING: bool = true;

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

    // let (mut graph, output_node) = build_network(inputs.size(), HL, 8);
    let (graph, value_node) = build_network(num_inputs, max_active, num_buckets, HL);

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        value_node,
        AdamWParams::default(),
        inputs,
        output_buckets,
        ["l0w", "l0b", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b"]
            .map(SavedFormat::id)
            .to_vec(),
        false,
    );

    let no_clipping = AdamWParams { min_weight: -128.0, max_weight: 128.0, ..AdamWParams::default() };

    trainer.optimiser_mut().set_params_for_weight("l2w", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l2b", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3w", no_clipping);
    trainer.optimiser_mut().set_params_for_weight("l3b", no_clipping);

    // trainer.load_from_checkpoint("checkpoints/scorpio-200");
    trainer.optimiser_mut().load_weights_from_file("delenda-b800-merged.raw").unwrap();

    let initial_lr;
    let final_lr;
    let sbs;
    if FINE_TUNING {
        initial_lr = 0.0005;
        final_lr = 0.0005 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;
        sbs = 200;
    } else {
        initial_lr = 0.001;
        final_lr = 0.001 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;
        sbs = 800;
    }

    let schedule = TrainingSchedule {
        net_id: "angel".into(),
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        eval_scale: 400.0,
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::LinearDecayLR { initial_lr, final_lr, final_superbatch: sbs },
            warmup_batches: 1600,
        },
        save_rate: 200,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/dataset.bin"]);
    // let data_loader = loader::ViriBinpackLoader::new(
    //     "data/dataset4.viriformat",
    //     4096,
    //     4,
    //     viriformat::dataformat::Filter {
    //         min_ply: 16,
    //         min_pieces: 4,
    //         max_eval: 31339,
    //         filter_tactical: true,
    //         filter_check: true,
    //         filter_castling: false,
    //         max_eval_incorrectness: u32::MAX,
    //     },
    // );

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}

fn build_network(num_inputs: usize, max_active: usize, output_buckets: usize, hl: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(output_buckets, 1), 1);

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, hl);
    let l1 = builder.new_affine("l1", hl, output_buckets * L2);
    let l2 = builder.new_affine("l2", L2, output_buckets * L3);
    let l3 = builder.new_affine("l3", L3, output_buckets);

    // 32 + 32 due to feature factoriser
    l0.init_with_effective_input_size(64);

    // inference
    let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
    let ntm_subnet = l0.forward(nstm).crelu().pairwise_mul();
    let accumulator = stm_subnet.concat(ntm_subnet);

    let l1_out = l1.forward(accumulator).select(buckets).screlu();
    let l2_out = l2.forward(l1_out).select(buckets).screlu();
    let l3_out = l3.forward(l2_out).select(buckets).sigmoid();

    l3_out.squared_error(targets);

    let value_node = l3_out.node();

    (builder.build(ExecutionContext::default()), value_node)
}
