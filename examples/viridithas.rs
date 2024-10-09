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

    let (mut graph, output_node) = build_network(inputs.size(), hl);

    graph.get_weights_mut("l0w").seed_random(0.0, 1.0 / (inputs.size() as f32).sqrt(), true);
    graph.get_weights_mut("l0b").seed_random(0.0, 1.0 / (inputs.size() as f32).sqrt(), true);
    graph.get_weights_mut("l1w").seed_random(0.0, 1.0 / (hl as f32).sqrt(), true);
    graph.get_weights_mut("l1b").seed_random(0.0, 1.0 / (hl as f32).sqrt(), true);
    graph.get_weights_mut("l2w").seed_random(0.0, 1.0 / 16f32.sqrt(), true);
    graph.get_weights_mut("l2b").seed_random(0.0, 1.0 / 16f32.sqrt(), true);
    graph.get_weights_mut("l3w").seed_random(0.0, 1.0 / 32f32.sqrt(), true);
    graph.get_weights_mut("l3b").seed_random(0.0, 1.0 / 32f32.sqrt(), true);

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::MaterialCount::<8>,
        vec![
            ("l0w".to_string(), QuantTarget::Float),
            ("l0b".to_string(), QuantTarget::Float),
            ("l1w".to_string(), QuantTarget::Float),
            ("l1b".to_string(), QuantTarget::Float),
            ("l2w".to_string(), QuantTarget::Float),
            ("l2b".to_string(), QuantTarget::Float),
            ("l3w".to_string(), QuantTarget::Float),
            ("l3b".to_string(), QuantTarget::Float),
        ],
    );

    let sbs = 800;
    let schedule = TrainingSchedule {
        net_id: "hyperstition".into(),
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        eval_scale: 400.0,
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3 * 0.3 * 0.3, final_superbatch: sbs },
            warmup_batches: 200,
        },
        save_rate: 200,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/dataset.bin"]);

    trainer.run(&schedule, &settings, &data_loader);
}

fn build_network(inputs: usize, hl: usize, output_buckets: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(inputs, 1));
    let nstm = builder.create_input("nstm", Shape::new(inputs, 1));
    let targets = builder.create_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0w = builder.create_weights("l0w", Shape::new(hl, inputs));
    let l0b = builder.create_weights("l0b", Shape::new(hl, 1));

    let l1w = builder.create_weights("l1w", Shape::new(16 * output_buckets, hl));
    let l1b = builder.create_weights("l1b", Shape::new(16 * output_buckets, 1));

    let l2w = builder.create_weights("l2w", Shape::new(32 * output_buckets, 16));
    let l2b = builder.create_weights("l2b", Shape::new(32 * output_buckets, 1));

    let l3w = builder.create_weights("l3w", Shape::new(1 * output_buckets, 32));
    let l3b = builder.create_weights("l3b", Shape::new(1 * output_buckets, 1));

    // inference
    let l1 = operations::sparse_affine_dual_with_activation(&mut builder, l0w, stm, nstm, l0b, Activation::CReLU);
    let paired = operations::pairwise_mul_post_sparse_affine_dual(&mut builder, l1);

    let l2 = operations::affine(&mut builder, l1w, paired, l1b);
    let l2 = operations::activate(&mut builder, l2, Activation::SCReLU);

    let l3 = operations::affine(&mut builder, l2w, l2, l2b);
    let l3 = operations::activate(&mut builder, l3, Activation::SCReLU);

    let predicted = operations::affine(&mut builder, l3w, l3, l3b);
    let sigmoided = operations::activate(&mut builder, predicted, Activation::Sigmoid);

    operations::mse(&mut builder, sigmoided, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
