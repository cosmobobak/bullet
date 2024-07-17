/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
There's potentially a lot of elo available by adjusting the wdl
and lr schedulers, depending on your dataset.
*/
use bullet_lib::{
    inputs, lr, optimiser, outputs, wdl, Activation, Loss, TrainerBuilder, TrainingSchedule,
};

const HIDDEN_SIZE: usize = 16;
const SCALE: i32 = 400;
const QA: i32 = 255;
const QB: i32 = 64;

fn main() {
    let trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .input(inputs::ChessBucketsMirrored::new([
            0, 1, 2, 3,
            4, 4, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            8, 8, 8, 8,
            8, 8, 8, 8,
            8, 8, 8, 8,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(1536)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "viridithas-annuep".into(),
        batch_size: 16_384,
        ft_regularisation: 0.0,
        eval_scale: 400.0,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 60,
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.001 * 0.3 * 0.3 * 0.3,
            final_superbatch: 60,
        },
        loss_function: Loss::SigmoidMSE,
        save_rate: 168,
        optimiser_settings: optimiser::AdamWParams {
            decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            min_weight: -1.98,
            max_weight: 1.98,
        },
    };

    trainer.load_from_checkpoint("viridithas-annuep-60");
    trainer.save("requant", schedule.net_id);
}

/*
This is how you would load the network in rust.
Commented out because it will error if it can't find the file.
static NNUE: Network =
    unsafe { std::mem::transmute(*include_bytes!("../checkpoints/simple-10/simple-10.bin")) };
*/

#[inline]
/// Clipped ReLU - Activation Function.
/// Note that this takes the i16s in the accumulator to i32s.
fn crelu(x: i16) -> i32 {
    i32::from(x).clamp(0, QA)
}

/// This is the quantised format that bullet outputs.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Column-Major `1 x (2 * HIDDEN_SIZE)`
    /// matrix, we use it like this to make the
    /// code nicer in `Network::evaluate`.
    output_weights: [i16; 2 * HIDDEN_SIZE],
    /// Scalar output bias.
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        // Initialise output with bias.
        let mut output = i32::from(self.output_bias);

        // Side-To-Move Accumulator -> Output.
        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            output += crelu(input) * i32::from(weight);
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            output += crelu(input) * i32::from(weight);
        }

        // Apply eval scale.
        output *= SCALE;

        // Remove quantisation.
        output /= QA * QB;

        output
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

impl Accumulator {
    /// Initialised with bias so we can just efficiently
    /// operate on it afterwards.
    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i -= *d
        }
    }
}