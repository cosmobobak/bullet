/*
Code to relabel a bulletformat dataset with a network
*/

use bullet_core::optimiser::utils::load_graph_weights_from_file;
use bullet_lib::{
    default::{inputs::ChessBucketsMirrored, outputs::Single}, nn::{Activation, ExecutionContext, Graph, NetworkBuilder, Node, Shape}, trainer::default::{
        formats::bulletformat::{ChessBoard, DataLoader},
        inputs,
        load_into_graph,
        loader::DefaultDataPreparer,
        outputs,
    }
};
use bulletformat::BulletFormat;
use std::{fs::File, io::BufWriter, time::Instant};

const NETWORK_PATH: &str = "checkpoints/sensei-800/optimiser_state/weights.bin";
const DATA_PATH: &str = "data/dataset.bin";
const OUTPUT_PATH: &str = "data/relabeled.bin";

// use bulletformat::ChessBoard;

const HL: usize = 8192;
const L2: usize = 16;
const L3: usize = 64;

type Input = ChessBucketsMirrored;
type Output = Single;

fn main() {
    #[rustfmt::skip]
    let inputs = Input::new([
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ]);
    let output_buckets = Output::default();

    let num_inputs = <Input as inputs::SparseInputType>::num_inputs(&inputs);
    let num_buckets = <Output as outputs::OutputBuckets<ChessBoard>>::BUCKETS;
    let batch_size = 16384;
    let eval_scale = 400.0;

    let (sender, receiver) = std::sync::mpsc::sync_channel(2);

    std::thread::spawn(move || {
        let loader = DataLoader::new(DATA_PATH, 128).unwrap();

        loader.map_batches(batch_size, |batch: &[ChessBoard]| {
            let prepared = DefaultDataPreparer::prepare(inputs, output_buckets, false, batch, 4, 0.0, eval_scale);
            sender.send((batch.to_vec(), prepared)).unwrap();
        });

        drop(sender);
    });

    let (sender2, receiver2) = std::sync::mpsc::sync_channel(2);

    std::thread::spawn(move || {
        let (mut graph, output_node) = build_network(num_inputs, num_buckets, HL);
        load_graph_weights_from_file::<ExecutionContext>(&mut graph, NETWORK_PATH, true).unwrap();

        let mut error = 0.0;
        let mut batches = 0;
        let mut positions = 0;
        let t = Instant::now();

        while let Ok((mut batch, prepared)) = receiver.recv() {
            unsafe {
                load_into_graph(&mut graph, &prepared).unwrap();
            }

            error += f64::from(graph.forward().unwrap());
            batches += 1;
            positions += batch.len();

            let scores = graph.get_node(output_node).get_dense_vals().unwrap();

            assert_eq!(batch.len(), scores.len());

            for (pos, result) in batch.iter_mut().zip(scores.iter()) {
                pos.score = (result * eval_scale).clamp(-32000.0, 32000.0) as i16;
            }

            sender2.send(batch).unwrap();

            if batches % 256 == 0 {
                let err = error / positions as f64;
                let pps = positions as f64 / t.elapsed().as_secs_f64() / 1000.0;
                println!("Avg Error: {err:.6}, Pos/Sec {pps:.1}k, total positions: {positions}");
            }
        }

        println!("Total Positions: {positions}");
    });

    let mut writer = BufWriter::new(File::create(OUTPUT_PATH).unwrap());
    while let Ok(batch) = receiver2.recv() {
        ChessBoard::write_to_bin(&mut writer, &batch).unwrap();
    }
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
