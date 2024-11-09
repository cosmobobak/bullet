use bullet_lib::{
    inputs::{self, InputType},
    loader, lr, operations,
    optimiser::{AdamWOptimiser, AdamWParams},
    outputs, wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, QuantTarget, Shape, Trainer,
    TrainingSchedule, TrainingSteps,
};
use diffable::Node;

fn main() {
    #[rustfmt::skip]
    let inputs = inputs::ChessBucketsMirroredFactorised::new([
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
         8,  9, 10, 11,
        12, 12, 13, 13,
        12, 12, 13, 13,
        14, 14, 15, 15,
        14, 14, 15, 15,
    ]);
    let hl = 2048;

    let fine_tuning = false;

    let (mut graph, output_node) = build_network(inputs.size(), hl, 8);

    graph.get_weights_mut("l0w").seed_random(0.0, 1.0 / (768f32).sqrt(), true);
    graph.get_weights_mut("l0b").seed_random(0.0, 1.0 / (768f32).sqrt(), true);
    graph.get_weights_mut("l1w").seed_random(0.0, 1.0 / (hl as f32).sqrt(), true);
    graph.get_weights_mut("l1b").seed_random(0.0, 1.0 / (hl as f32).sqrt(), true);
    graph.get_weights_mut("l1skipw").seed_random(0.0, 1.0 / (hl as f32).sqrt(), true);
    graph.get_weights_mut("l2w").seed_random(0.0, 1.0 / 15f32.sqrt(), true);
    graph.get_weights_mut("l2b").seed_random(0.0, 1.0 / 15f32.sqrt(), true);
    graph.get_weights_mut("l3w").seed_random(0.0, 1.0 / 32f32.sqrt(), true);
    graph.get_weights_mut("psqtw").seed_random(0.0, 1.0 / 768f32.sqrt(), true);
    graph.get_weights_mut("outb").seed_random(0.0, 1.0 / 32f32.sqrt(), true);

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::MaterialCount::<8>,
        vec![
            ("l0w".to_string(), QuantTarget::Float),
            ("l0b".to_string(), QuantTarget::Float),
            // emit l1 / l1skip contiguously so that reinterpretation is easy
            ("l1w".to_string(), QuantTarget::Float),
            ("l1skipw".to_string(), QuantTarget::Float),
            ("l1b".to_string(), QuantTarget::Float),
            // embed this here, gets added automagically.
            ("outb".to_string(), QuantTarget::Float),
            ("l2w".to_string(), QuantTarget::Float),
            ("l2b".to_string(), QuantTarget::Float),
            ("l3w".to_string(), QuantTarget::Float),
            ("psqtw".to_string(), QuantTarget::Float),
        ],
    );

    //trainer.load_from_checkpoint("checkpoints/voyager-800");

    let initial_lr;
    let final_lr;
    let sbs;
    if fine_tuning {
        initial_lr = 0.0005;
        final_lr = 0.0005 * 0.3 * 0.3 * 0.3;
        sbs = 100;
    } else {
        initial_lr = 0.001;
        final_lr = 0.001 * 0.3 * 0.3 * 0.3;
        sbs = 800;
    }

    let schedule = TrainingSchedule {
        net_id: "nostromo".into(),
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        eval_scale: 400.0,
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: sbs },
            warmup_batches: 200,
        },
        save_rate: 200,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/dataset.bin"]);

    trainer.run(&schedule, &settings, &data_loader);
}

fn build_network(inputs: usize, hl: usize, output_buckets: usize) -> (Graph, Node) {
    let builder = &mut GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(inputs, 1));
    let nstm = builder.create_input("nstm", Shape::new(inputs, 1));
    let targets = builder.create_input("targets", Shape::new(1, 1));
    let buckets = builder.create_input("buckets", Shape::new(output_buckets, 1));

    // trainable weights
    let l0w = builder.create_weights("l0w", Shape::new(hl, inputs));
    let l0b = builder.create_weights("l0b", Shape::new(hl, 1));

    let l1w = builder.create_weights("l1w", Shape::new(15 * output_buckets, hl));
    let l1b = builder.create_weights("l1b", Shape::new(15 * output_buckets, 1));

    // skip connection, adding a dotprod of the accumulator direct to the output.
    let l1skipw = builder.create_weights("l1skipw", Shape::new(output_buckets, hl));

    let l2w = builder.create_weights("l2w", Shape::new(32 * output_buckets, 15));
    let l2b = builder.create_weights("l2b", Shape::new(32 * output_buckets, 1));

    let l3w = builder.create_weights("l3w", Shape::new(output_buckets, 32));

    let psqt = builder.create_weights("psqtw", Shape::new(output_buckets, inputs));

    // biases shared by each layer that gets added into the output:
    let output_bias = builder.create_weights("outb", Shape::new(output_buckets, 1));

    // inference
    let l1 = operations::sparse_affine_dual_with_activation(builder, l0w, stm, nstm, l0b, Activation::CReLU);
    let paired = operations::pairwise_mul_post_sparse_affine_dual(builder, l1);

    let l2 = operations::affine(builder, l1w, paired, l1b);
    let l2 = operations::select(builder, l2, buckets);
    let l2 = operations::activate(builder, l2, Activation::SCReLU);

    let l3 = operations::affine(builder, l2w, l2, l2b);
    let l3 = operations::select(builder, l3, buckets);
    let l3 = operations::activate(builder, l3, Activation::SCReLU);

    let l1skip_out = operations::affine(builder, l1skipw, paired, output_bias);
    let l1skip_out = operations::select(builder, l1skip_out, buckets);

    let main_net_out = operations::affine(builder, l3w, l3, output_bias);
    let main_net_out = operations::select(builder, main_net_out, buckets);

    let main_net_out = operations::add(builder, main_net_out, l1skip_out);

    let psqt_out = operations::affine(builder, psqt, stm, output_bias);
    let psqt_out = operations::select(builder, psqt_out, buckets);

    let predicted = operations::add(builder, main_net_out, psqt_out);

    let sigmoided = operations::activate(builder, predicted, Activation::Sigmoid);

    operations::mse(builder, sigmoided, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
