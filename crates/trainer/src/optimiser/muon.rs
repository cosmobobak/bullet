//! Muon: “**M**oment**U**m **O**rthogonalized by **N**ewton-schulz”.
//!
//! <https://kellerjordan.github.io/posts/muon/>
//!
//! Muon runs SGD-momentum and then replaces each 2D parameter’s update
//! with the nearest (semi-)orthogonal matrix, computed with a quintic
//! Newton-Schulz iteration that can be run stably on the GPU.
//!
//! Muon only works on ≥2-dimensional parameter buffers, and so delegates
//! to another optimiers for 1-dimensional parameters. In addition:
//!
//! > Muon is designed to update matrix-based parameters. In practice,
//! > AdamW is used in couple with Muon to handle non-matrix based
//! > parameters, like RMSNorm, LM head, and embedding parameters.
//! > — Muon is Scalable for LLM Training (<https://arxiv.org/abs/2502.16982>)
//!
//! As layer 0 of NNUEs (the “feature transformer”) is much like an
//! embedding matrix, it may make sense to train it with AdamW over
//! Muon.
//!
//! Each Muon update is performed in three stages:
//! 1. a pointwise kernel updates the momentum buffer and forms the raw update,
//! 2. a [`Function`] runs the Newton-Schulz orthogonalisation, and
//! 3. a pointwise kernel applies weight decay and the orthogonalised update to the weights.

use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    model::Shape,
    tensor::{
        DType, DValue, IRBuilder, IRTrace, TNode, TType, TValue,
        operation::{CABinary, Matmul, MatrixLayout},
    },
};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseIR,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::optimiser::{
    OptimiserState, OptimiserUpdateResult, OptimiserUpdateSync,
    adam::{AdamW, AdamWParams},
    utils,
};

/// Verbatim from <https://kellerjordan.github.io/posts/muon/>.
const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

#[derive(Clone, Copy, Debug)]
pub struct MuonParams {
    /// Momentum coefficient
    pub momentum: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Number of Newton-Schulz iterations
    pub ns_steps: usize,

    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for MuonParams {
    fn default() -> Self {
        Self { momentum: 0.95, weight_decay: 0.0, ns_steps: 5, min_weight: -1.98, max_weight: 1.98 }
    }
}

/// Parameters for [`MuonWithAuxAdam`].
#[derive(Clone, Copy, Debug)]
pub struct MuonWithAuxAdamWParams {
    pub use_muon: bool,
    // TODO: reevaluate
    pub muon: MuonParams,
    pub adam: AdamWParams,
}

impl Default for MuonWithAuxAdamWParams {
    fn default() -> Self {
        Self { use_muon: true, muon: MuonParams::default(), adam: AdamWParams::default() }
    }
}

impl MuonWithAuxAdamWParams {
    fn muon_eligible(&self, shape: Shape) -> bool {
        self.use_muon && shape.rows() > 1 && shape.cols() > 1
    }
}

/// pointwise kernel for momentum update + creation of raw update.
///
/// `momentum ← momentum.lerp(g, 1 - beta)` and `u₀ ← g.lerp(momentum, beta)`,
/// where `g = gradient_factor * gradient`.
fn build_pre_op(size: usize, beta: f32, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let mut p = PointwiseIR::new(size.into())?;

    // constants for zero, beta, and 1 - beta.
    let zero = p.add_const(DValue::I32(0), 0);
    let beta_c = p.add_const(DValue::F32(beta), 0);
    let omb_c = p.add_const(DValue::F32(1.0 - beta), 0);

    // `gf` is the batch_size normalisation for gradients.
    let gf = p.add_buf(TType::new(1, DType::F32));
    // the gradient!
    let grad = p.add_buf(TType::new(size, DType::F32));

    // in-out buffer for momentum
    let mom = p.add_buf(TType::new(size, DType::F32));

    // out buffer for raw update
    let u0 = p.add_buf(TType::new(size, DType::F32));

    // scale gradients by grad factor:
    let gf_v = p.read(gf, zero, 0)?;
    let g = p.read(grad, p.tid(), 0)?;
    let g = p.binary(g, gf_v, CABinary::Mul)?;

    // update momentum:
    let m_old = p.read(mom, p.tid(), 0)?;
    // m_new = beta × m_old + (1 - beta) × g
    let lhs = p.binary(beta_c, m_old, CABinary::Mul)?;
    let rhs = p.binary(omb_c, g, CABinary::Mul)?;
    let m_new = p.binary(lhs, rhs, CABinary::Add)?;
    p.write(mom, p.tid(), m_new)?;

    // generate u₀
    // u₀ = (1 - beta) × g + beta × m_new
    let lhs = p.binary(omb_c, g, CABinary::Mul)?;
    let rhs = p.binary(beta_c, m_new, CABinary::Mul)?;
    let u = p.binary(lhs, rhs, CABinary::Add)?;
    p.write(u0, p.tid(), u)?;

    unsafe { p.lower("muon_pre".to_string(), props) }.map_err(Into::into)
}

/// pointwise kernel for decoupled weight decay,
/// applying the update, and clamping the result.
///
/// `weight <- clamp(weight * (1 - lr * decay) - lr * u, min, max)`.
fn build_post_op(size: usize, decay: f32, min: f32, max: f32, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let mut p = PointwiseIR::new(size.into())?;

    // constants for 0, 1, –1, decay, min, max
    let zero = p.add_const(DValue::I32(0), 0);
    let one = p.add_const(DValue::F32(1.0), 0);
    let neg = p.add_const(DValue::F32(-1.0), 0);
    let decay_c = p.add_const(DValue::F32(decay), 0);
    let min_c = p.add_const(DValue::F32(min), 0);
    let max_c = p.add_const(DValue::F32(max), 0);

    // learning rate
    let lr = p.add_buf(TType::new(1, DType::F32));
    // orthogonalised gradient matrix
    let u = p.add_buf(TType::new(size, DType::F32));
    // read-write weights buffer
    let w = p.add_buf(TType::new(size, DType::F32));

    let lr_v = p.read(lr, zero, 0)?;

    // do decoupled weight decay:
    // scale decay, amount = lr × decay
    let amt = p.binary(lr_v, decay_c, CABinary::Mul)?;
    // compute –amount
    let neg_amt = p.binary(neg, amt, CABinary::Mul)?;
    // factor = 1 - amount
    let factor = p.binary(one, neg_amt, CABinary::Add)?;
    // the above calculation is a little weird
    // if decay is e.g. 0.01 and LR is 0.001,
    // then factor will be
    //   1 – 0.001 × 0.01
    // = 1 – 0.00001
    // = 0.99999

    // apply decay
    let w_old = p.read(w, p.tid(), 0)?;
    let decayed = p.binary(w_old, factor, CABinary::Mul)?;

    // compute the diff to apply to the weights:
    // d = –(lr × u)
    let uu = p.read(u, p.tid(), 0)?;
    let lru = p.binary(lr_v, uu, CABinary::Mul)?;
    let neg_lru = p.binary(neg, lru, CABinary::Mul)?;

    // get new weights by applying the
    // gradient to the decay’d weights
    let w_new = p.binary(decayed, neg_lru, CABinary::Add)?;

    // clamp(w_new, min, max)
    let clamped = p.binary(w_new, max_c, CABinary::Min)?;
    let clamped = p.binary(clamped, min_c, CABinary::Max)?;
    p.write(w, p.tid(), clamped)?;

    unsafe { p.lower("muon_post".to_string(), props) }.map_err(Into::into)
}

/// Newton-Schulz orthogonalisation as a `Function`.
///
/// in: raw update `U₀` (flat buffer of `rows × cols`).
/// out: orthogonalised and scaled update.
fn build_ns_function<G: Gpu>(
    device: &Arc<Device<G>>,
    shape: Shape,
    steps: usize,
) -> Result<(Function<G>, NodeId, NodeId), G::Error> {
    let g = IRBuilder::default();
    let (x_in, x_out) = build_ns_graph(&g, shape, steps).map_err(|e| format!("Failed to build Muon NS graph: {e}"))?;
    let (in_id, out_id) = (x_in.node(), x_out.node());

    let ir = g.build([x_out]);
    let mut func = Function::new(device.clone(), ir).map_err(|e| format!("Failed to compile Muon NS graph: {e}"))?;
    func.prealloc()?;

    Ok((func, in_id, out_id))
}

/// constructs the Newton-Schulz function and returns its input and output nodes.
/// this function is adapted from the following Pytorch code:
///
/// ```py
/// def newtonschulz5(G, steps=5, eps=1e-7):
///     assert G.ndim == 2
///     a, b, c = (3.4445, -4.7750, 2.0315)
///     X = G.bfloat16()
///     X /= (X.norm() + eps)
///     if G.size(0) > G.size(1):
///         X = X.T
///     for _ in range(steps):
///         A = X @ X.T
///         B = b * A + c * A @ A
///         X = a * X + B @ X
///     if G.size(0) > G.size(1):
///         X = X.T
///     return X
/// ```
fn build_ns_graph(g: &IRBuilder, shape: Shape, ns_steps: usize) -> Result<(TNode<'_>, TNode<'_>), IRTrace> {
    let (rows, cols) = (shape.rows(), shape.cols());
    let n = shape.size();
    let (a, b_coeff, c_coeff) = NS_COEFFS;

    let x_in = g.add_input(n, DType::F32);

    // X.norm() in Pytorch is the same as torch.linalg.matrix_norm,
    // which (by default) applies something called the Frobenius
    // norm. See <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>.
    // Thankfully, it appears to literally just be the l₂ norm of
    // the flattened matrix.
    let sq = (x_in * x_in)?;
    let sum_sq = sq.reduce_sum([1usize, n], 1)?;
    let norm = sum_sq.sqrt()?;

    // scale the matrix:
    let denom = (norm + 1e-7f32)?;
    let inv = (1.0f32 / denom)?;
    let inv = inv.broadcast([1usize, 1usize], 1, n)?;
    let mut x = (x_in * inv)?;

    // reïnterpret the buffer as a proper matrix
    let lx = MatrixLayout { rows: rows.into(), cols: cols.into(), col_mjr: true };

    // do iterations of NS
    for _ in 0..ns_steps {
        // the Pytorch listing transposes and then
        // untransposes matrixes when rows > cols.
        // we instead just do it two different ways.
        //
        // desired routine:
        // A = X @ X.T
        // B = b * A + c * A @ A
        // X = a * X + B @ X
        if rows <= cols {
            let la = MatrixLayout { rows: rows.into(), cols: rows.into(), col_mjr: true };
            // A = X @ Xᵀ
            let amat = g.add_op([x, x], Matmul::new(DType::F32, 1, lx, lx.transpose())?)?[0];
            // A₂ = A @ A
            let a2 = g.add_op([amat, amat], Matmul::new(DType::F32, 1, la, la)?)?[0];
            // B = b·A + c·A₂
            let bmat = ((amat * b_coeff)? + (a2 * c_coeff)?)?;
            // X ← a·X + B @ X
            let bx = g.add_op([bmat, x], Matmul::new(DType::F32, 1, la, lx)?)?[0];
            x = ((x * a)? + bx)?;
        } else {
            let lc = MatrixLayout { rows: cols.into(), cols: cols.into(), col_mjr: true };
            // A = Xᵀ @ X
            let amat = g.add_op([x, x], Matmul::new(DType::F32, 1, lx.transpose(), lx)?)?[0];
            // A₂ = A @ A
            let a2 = g.add_op([amat, amat], Matmul::new(DType::F32, 1, lc, lc)?)?[0];
            // B = b·A + c·A₂
            let bmat = ((amat * b_coeff)? + (a2 * c_coeff)?)?;
            // X ← a·X + X B
            let xb = g.add_op([x, bmat], Matmul::new(DType::F32, 1, lx, lc)?)?[0];
            x = ((x * a)? + xb)?;
        }
    }

    // Scale to account for the change in spectral norm, sqrt(max(1, rows/cols)).
    // taken from the Pytorch Muon implementation at
    // <https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py#L40>
    let scale = (rows as f32 / cols as f32).max(1.0).sqrt();
    let x = (x * scale)?;

    Ok((x_in, x))
}

struct MuonState<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    raw_update: Arc<Buffer<G>>,
    ortho_update: Arc<Buffer<G>>,
    pre: CompiledKernel<G>,
    post: CompiledKernel<G>,
    ns: Function<G>,
    ns_in: NodeId,
    ns_out: NodeId,
}

impl<G: Gpu> MuonState<G> {
    fn new(device: &Arc<Device<G>>, shape: Shape, params: MuonParams) -> Result<Self, G::Error> {
        let size = shape.size();

        if params.max_weight < params.min_weight {
            return Err(format!("Invalid clipping: {} >= {}", params.min_weight, params.max_weight).into());
        }

        let pre = build_pre_op(size, params.momentum, device.props()).unwrap().compile(device.clone())?;
        let post = build_post_op(size, params.weight_decay, params.min_weight, params.max_weight, device.props())
            .unwrap()
            .compile(device.clone())?;
        let (ns, ns_in, ns_out) = build_ns_function(device, shape, params.ns_steps)?;

        Ok(Self {
            momentum: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            raw_update: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            ortho_update: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            pre,
            post,
            ns,
            ns_in,
            ns_out,
        })
    }
}

pub struct MuonWithAuxAdamW<G: Gpu> {
    device: Arc<Device<G>>,
    shape: Shape,
    inner: Inner<G>,
}

enum Inner<G: Gpu> {
    Muon(Box<MuonState<G>>),
    Adam(Box<AdamW<G>>),
}

impl<G: Gpu> OptimiserState<G> for MuonWithAuxAdamW<G> {
    type Params = MuonWithAuxAdamWParams;

    fn new(device: &Arc<Device<G>>, shape: Shape, params: Self::Params) -> Result<Self, G::Error> {
        let inner = if params.muon_eligible(shape) {
            Inner::Muon(Box::new(MuonState::new(device, shape, params.muon)?))
        } else {
            Inner::Adam(Box::new(<AdamW<G> as OptimiserState<G>>::new(device, shape, params.adam)?))
        };

        Ok(Self { device: device.clone(), shape, inner })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        match &mut self.inner {
            Inner::Adam(adam) => adam.update(stream, weights, grads, gradient_factor, learning_rate),
            Inner::Muon(state) => {
                let mut sync = OptimiserUpdateSync::default();

                // momentum & raw update
                sync.push_kernel(state.pre.execute(
                    stream.clone(),
                    vec![gradient_factor, grads],
                    vec![state.momentum.clone(), state.raw_update.clone()],
                )?);

                // ns
                let mut io = BTreeMap::new();
                io.insert(state.ns_in, state.raw_update.clone());
                io.insert(state.ns_out, state.ortho_update.clone());
                // TODO: I don’t know what I’m doing with this bit.
                state.ns.execute(stream.clone(), &io)?.value()?;

                // decay & update
                sync.push_kernel(state.post.execute(
                    stream.clone(),
                    vec![learning_rate, state.ortho_update.clone()],
                    vec![weights],
                )?);

                Ok(sync)
            }
        }
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        match &mut self.inner {
            Inner::Adam(adam) => adam.reset(),
            Inner::Muon(state) => {
                let size = state.momentum.size();
                state.momentum.copy_from_host(&TValue::zeros(DType::F32, size))?;
                state.raw_update.copy_from_host(&TValue::zeros(DType::F32, size))?;
                state.ortho_update.copy_from_host(&TValue::zeros(DType::F32, size))?;
                Ok(())
            }
        }
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        let device = self.device.clone();
        let shape = self.shape;
        *self = Self::new(&device, shape, params)?;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let muon: Vec<_> = map
            .iter()
            .filter_map(|(id, single)| match &single.inner {
                Inner::Muon(state) => Some((id, &state.momentum)),
                Inner::Adam(_) => None,
            })
            .collect();
        utils::write_weights_to_file::<G>(&muon, &format!("{path}/muon_momentum.bin"))?;

        let adam: BTreeMap<String, &AdamW<G>> = map
            .iter()
            .filter_map(|(id, single)| match &single.inner {
                Inner::Adam(adam) => Some((id.clone(), &**adam)),
                Inner::Muon(_) => None,
            })
            .collect();
        AdamW::write_to_checkpoint(&adam, path)
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        for (id, mom) in utils::load_weights_from_file(&format!("{path}/muon_momentum.bin")) {
            if let Some(single) = map.get_mut(&id) {
                if let Inner::Muon(state) = &mut single.inner {
                    state.momentum.copy_from_host(&TValue::F32(mom))?;
                }
            }
        }

        let mut adam: BTreeMap<String, &mut AdamW<G>> = map
            .iter_mut()
            .filter_map(|(id, single)| match &mut single.inner {
                Inner::Adam(adam) => Some((id.clone(), &mut **adam)),
                Inner::Muon(_) => None,
            })
            .collect();

        AdamW::load_from_checkpoint(&mut adam, path)
    }
}
