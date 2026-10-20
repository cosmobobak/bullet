//! Muon: “**M**oment**U**m **O**rthogonalized by **N**ewton-schulz”.
//!
//! <https://kellerjordan.github.io/posts/muon/>
//!
//! Muon runs SGD-momentum and then replaces each 2D parameter’s update
//! with the nearest (semi-)orthogonal matrix, computed with a quintic
//! Newton-Schulz iteration that can be run stably on the GPU.
//!
//! Muon only works on ≥2-dimensional parameter buffers, and so delegates
//! to AdamW for 1-dimensional parameters. In addition:
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
        DType, IRBuilder, IRTrace, TNode, TType, TValue,
        operation::{Matmul, MatrixLayout},
    },
};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
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
pub struct MuonMatrixParams {
    /// Momentum coefficient
    pub momentum: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Number of Newton-Schulz iterations
    pub ns_steps: usize,

    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for MuonMatrixParams {
    fn default() -> Self {
        Self { momentum: 0.95, weight_decay: 0.0, ns_steps: 5, min_weight: -1.98, max_weight: 1.98 }
    }
}

/// Parameters for [`Muon`].
#[derive(Clone, Copy, Debug)]
pub struct MuonParams {
    pub use_muon: bool,
    pub muon: MuonMatrixParams,
    pub adam: AdamWParams,
}

impl Default for MuonParams {
    fn default() -> Self {
        Self { use_muon: true, muon: MuonMatrixParams::default(), adam: AdamWParams::default() }
    }
}

impl MuonParams {
    fn muon_eligible(&self, shape: Shape) -> bool {
        self.use_muon && shape.rows() > 1 && shape.cols() > 1
    }
}

/// `momentum ← momentum.lerp(g, 1 - beta)` and `u₀ ← g.lerp(momentum, beta)`,
/// where `g = gradient_factor * gradient`.
fn build_momentum_op(size: usize, beta: f32, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let (builder, p2size) = PointwiseBuilder::vectorised(size);

    let ty = TType::new(size, DType::F32);

    let adj_buf = builder.new_buffer(TType::new(1, DType::F32));
    let grad_buf = builder.new_buffer(ty);
    let momentum_buf = builder.new_buffer(ty);
    let update_buf = builder.new_buffer(ty);

    let adj = adj_buf.read(0, 0).splat(p2size);

    let tid = builder.tid();
    let grad = adj * grad_buf.read(tid, p2size);
    let momentum = beta * momentum_buf.read(tid, p2size) + (1.0 - beta) * grad;
    let update = (1.0 - beta) * grad + beta * momentum;

    momentum_buf.write(tid, momentum);
    update_buf.write(tid, update);

    builder.ir().eliminate_common_subexprs()?;

    unsafe { Ok(builder.inner().lower("muon_momentum".to_string(), props)?) }
}

/// `weight <- clamp(weight * (1 - lr * decay) - lr * u, min, max)`.
fn build_apply_op(size: usize, params: &MuonMatrixParams, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let (builder, p2size) = PointwiseBuilder::vectorised(size);

    let ty = TType::new(size, DType::F32);

    let rate_buf = builder.new_buffer(TType::new(1, DType::F32));
    let update_buf = builder.new_buffer(ty);
    let weight_buf = builder.new_buffer(ty);

    let rate = rate_buf.read(0, 0).splat(p2size);

    let tid = builder.tid();
    let weight = weight_buf.read(tid, p2size) * (1.0 - params.weight_decay * rate);
    let weight = weight - rate * update_buf.read(tid, p2size);

    weight_buf.write(tid, weight.max(params.min_weight).min(params.max_weight));

    builder.ir().eliminate_common_subexprs()?;

    unsafe { Ok(builder.inner().lower("muon_apply".to_string(), props)?) }
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
fn build_ns_graph<'a>(g: &'a IRBuilder, shape: Shape, ns_steps: usize) -> Result<(TNode<'a>, TNode<'a>), IRTrace> {
    let (rows, cols) = (shape.rows(), shape.cols());
    let n = shape.size();
    let (a, b, c) = NS_COEFFS;

    let x_in = g.add_input(n, DType::F32);

    // X.norm() in Pytorch is the same as torch.linalg.matrix_norm,
    // which (by default) applies something called the Frobenius
    // norm. See <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>.
    // Thankfully, it appears to literally just be the l₂ norm of
    // the flattened matrix.
    let norm = (x_in * x_in)?.reduce_sum([1usize, n], 1)?.sqrt()?;
    let inv = (1.0f32 / (norm + 1e-7f32)?)?.broadcast([1usize, 1usize], 1, n)?;
    let mut x = (x_in * inv)?;

    let matmul = |lhs: TNode<'a>, rhs: TNode<'a>, llayout, rlayout| -> Result<_, IRTrace> {
        Ok(g.add_op([lhs, rhs], Matmul::new(DType::F32, 1, llayout, rlayout)?)?[0])
    };

    // the Pytorch listing transposes and then untransposes the
    // matrix when rows > cols, we instead form the gram matrix
    // on the smaller side and multiply from the other side.
    let transposed = rows > cols;
    let k = rows.min(cols);
    let lx = MatrixLayout { rows: rows.into(), cols: cols.into(), col_mjr: true };
    let lk = MatrixLayout { rows: k.into(), cols: k.into(), col_mjr: true };

    // desired routine:
    // A = X @ X.T
    // B = b * A + c * A @ A
    // X = a * X + B @ X
    for _ in 0..ns_steps {
        let amat = if transposed { matmul(x, x, lx.transpose(), lx)? } else { matmul(x, x, lx, lx.transpose())? };
        let a2 = matmul(amat, amat, lk, lk)?;
        let bmat = ((amat * b)? + (a2 * c)?)?;
        let bx = if transposed { matmul(x, bmat, lx, lk)? } else { matmul(bmat, x, lk, lx)? };
        x = ((x * a)? + bx)?;
    }

    // The update here must be scaled. In the original Pytorch reference implementation,
    // (found at <https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py#L40>)
    // scale was computed as scale = sqrt(max(1, rows / cols)). However, in Moonshot’s
    // “Muon is Scalable for LLM Training” §2.2 Consistent update RMS, they say the following:
    //
    // > An important property of Adam and AdamW is that they maintain a theoretical update
    // > RMS around 1. However, we show that Muon’s update RMS varies depending on the shape
    // > parameters, according to the following lemma:
    // > Lemma 1. For a full rank matrix parameter of shape [A, B],
    // >          its theoretical Muon update RMS is sqrt(1/max(A, B))
    // > […] We monitored Muon’s update RMS during training and found it typically close to
    // > the theoretical value given above. […] In order to maintain consistent update RMS
    // > among matrices of different shapes, we propose to scale the Muon update for each
    // > matrix by its sqrt(max(A, B)) to cancel the effects of Lemma 1.
    //
    // Later, in §2.2 Matching update RMS of AdamW:
    //
    // > From empirical observations, AdamW’s update RMS is usually around 0.2 to 0.4.
    // > Therefore, we scale Muon’s update RMS to the range by the following adjustment:
    // >   W_t = W_{t-1} – η_t (0.2 · O_t · sqrt(max(A,B)) + λW_{t-1})
    // > We validated this choice with empirical results […]. Moreover, we highlighted
    // > that with this adjustment, Muon can directly *reuse* the learning rate and weight
    // > decay tuned for AdamW.
    //
    // Link to paper: <https://arxiv.org/pdf/2502.16982>
    //
    // Hence:
    let scale = 0.2 * (rows.max(cols) as f32).sqrt();
    let x = (x * scale)?;

    Ok((x_in, x))
}

struct MuonOps<G: Gpu> {
    momentum: CompiledKernel<G>,
    apply: CompiledKernel<G>,
    ns: Function<G>,
    ns_in: NodeId,
    ns_out: NodeId,
}

impl<G: Gpu> MuonOps<G> {
    fn new(device: &Arc<Device<G>>, shape: Shape, params: MuonMatrixParams) -> Result<Self, G::Error> {
        if params.max_weight < params.min_weight {
            return Err(format!("Invalid clipping: {} >= {}", params.min_weight, params.max_weight).into());
        }

        let size = shape.size();
        let momentum = build_momentum_op(size, params.momentum, device.props()).unwrap().compile(device.clone())?;
        let apply = build_apply_op(size, &params, device.props()).unwrap().compile(device.clone())?;
        let (ns, ns_in, ns_out) = build_ns_function(device, shape, params.ns_steps)?;

        Ok(Self { momentum, apply, ns, ns_in, ns_out })
    }
}

struct MuonState<G: Gpu> {
    momentum: Arc<Buffer<G>>,
    raw_update: Arc<Buffer<G>>,
    ortho_update: Arc<Buffer<G>>,
    ops: MuonOps<G>,
}

impl<G: Gpu> MuonState<G> {
    fn new(device: &Arc<Device<G>>, shape: Shape, params: MuonMatrixParams) -> Result<Self, G::Error> {
        let zeros = || Buffer::from_host(device, &TValue::zeros(DType::F32, shape.size()));

        Ok(Self {
            momentum: zeros()?,
            raw_update: zeros()?,
            ortho_update: zeros()?,
            ops: MuonOps::new(device, shape, params)?,
        })
    }

    fn update<'a>(
        &'a self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        let mut sync = OptimiserUpdateSync::default();

        sync.push_kernel(self.ops.momentum.execute(
            stream.clone(),
            vec![gradient_factor, grads],
            vec![self.momentum.clone(), self.raw_update.clone()],
        )?);

        let io =
            BTreeMap::from([(self.ops.ns_in, self.raw_update.clone()), (self.ops.ns_out, self.ortho_update.clone())]);
        sync.push_function(self.ops.ns.execute(stream.clone(), &io)?);

        sync.push_kernel(self.ops.apply.execute(
            stream.clone(),
            vec![learning_rate, self.ortho_update.clone()],
            vec![weights],
        )?);

        Ok(sync)
    }

    fn set_params(&mut self, device: &Arc<Device<G>>, shape: Shape, params: MuonMatrixParams) -> Result<(), G::Error> {
        self.ops.ns.dealloc_preallocs()?;
        self.ops = MuonOps::new(device, shape, params)?;
        Ok(())
    }
}

pub struct Muon<G: Gpu> {
    device: Arc<Device<G>>,
    shape: Shape,
    inner: Inner<G>,
}

enum Inner<G: Gpu> {
    Muon(Box<MuonState<G>>),
    Adam(Box<AdamW<G>>),
    Freed,
}

impl<G: Gpu> Inner<G> {
    fn new(device: &Arc<Device<G>>, shape: Shape, params: MuonParams) -> Result<Self, G::Error> {
        Ok(if params.muon_eligible(shape) {
            Self::Muon(Box::new(MuonState::new(device, shape, params.muon)?))
        } else {
            Self::Adam(Box::new(<AdamW<G> as OptimiserState<G>>::new(device, shape, params.adam)?))
        })
    }
}

impl<G: Gpu> OptimiserState<G> for Muon<G> {
    type Params = MuonParams;

    fn new(device: &Arc<Device<G>>, shape: Shape, params: Self::Params) -> Result<Self, G::Error> {
        Ok(Self { device: device.clone(), shape, inner: Inner::new(device, shape, params)? })
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
            Inner::Muon(state) => state.update(stream, weights, grads, gradient_factor, learning_rate),
            Inner::Freed => unreachable!(),
        }
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        match &mut self.inner {
            Inner::Adam(adam) => adam.reset(),
            Inner::Muon(state) => state.momentum.copy_from_host(&TValue::zeros(DType::F32, state.momentum.size())),
            Inner::Freed => unreachable!(),
        }
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        match (&mut self.inner, params.muon_eligible(self.shape)) {
            (Inner::Muon(state), true) => state.set_params(&self.device, self.shape, params.muon),
            (Inner::Adam(adam), false) => adam.set_params(params.adam),
            _ => {
                self.inner = Inner::Freed;
                self.inner = Inner::new(&self.device, self.shape, params)?;
                Ok(())
            }
        }
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let muon: Vec<_> = map
            .iter()
            .filter_map(|(id, single)| match &single.inner {
                Inner::Muon(state) => Some((id, &state.momentum)),
                _ => None,
            })
            .collect();
        utils::write_weights_to_file::<G>(&muon, &format!("{path}/muon_momentum.bin"))?;

        let adam = map
            .iter()
            .filter_map(|(id, single)| match &single.inner {
                Inner::Adam(adam) => Some((id.clone(), &**adam)),
                _ => None,
            })
            .collect();
        AdamW::write_to_checkpoint(&adam, path)
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        for (id, momentum) in utils::load_weights_from_file(&format!("{path}/muon_momentum.bin")) {
            if let Some(Inner::Muon(state)) = map.get_mut(&id).map(|single| &mut single.inner) {
                state.momentum.copy_from_host(&TValue::F32(momentum))?;
            }
        }

        let mut adam = map
            .iter_mut()
            .filter_map(|(id, single)| match &mut single.inner {
                Inner::Adam(adam) => Some((id.clone(), &mut **adam)),
                _ => None,
            })
            .collect();
        AdamW::load_from_checkpoint(&mut adam, path)
    }
}


