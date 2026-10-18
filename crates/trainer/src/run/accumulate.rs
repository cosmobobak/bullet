use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::{DType, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    pointwise::PointwiseBuilder,
    runtime::{Device, DeviceProps, Gpu, Stream},
};

use crate::{model::TensorMap, optimiser::OptimiserUpdateSync};

/// Implements gradient accumulation so we can have batch
/// sizes greater than GPU memory would ordinarily allow.
pub struct GradientAccumulator<G: Gpu> {
    sums: TensorMap<G>,
    kernels: BTreeMap<String, CompiledKernel<G>>,
    zero: Arc<Buffer<G>>,
    one: Arc<Buffer<G>>,
}

fn build(size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let (builder, p2size) = PointwiseBuilder::vectorised(size);

    let ty = TType::new(size, DType::F32);

    let keep_buf = builder.new_buffer(TType::new(1, DType::F32));
    let grad_buf = builder.new_buffer(ty);
    let sum_buf = builder.new_buffer(ty);

    let keep = keep_buf.read(0, 0).splat(p2size);

    let tid = builder.tid();
    let sum = keep * sum_buf.read(tid, p2size) + grad_buf.read(tid, p2size);
    sum_buf.write(tid, sum);

    builder.ir().eliminate_common_subexprs()?;

    unsafe { Ok(builder.inner().lower("accumulate".to_string(), props)?) }
}

impl<G: Gpu> GradientAccumulator<G> {
    pub fn new(device: &Arc<Device<G>>, gradients: &TensorMap<G>, steps: usize) -> Result<Self, G::Error> {
        let zero = Buffer::from_host(device, &TValue::F32(vec![0.0]))?;
        let one = Buffer::from_host(device, &TValue::F32(vec![1.0]))?;

        // summing over a single batch is a nop
        if steps == 1 {
            return Ok(Self { sums: gradients.clone(), kernels: BTreeMap::new(), zero, one });
        }

        let mut sums = BTreeMap::new();
        let mut kernels = BTreeMap::new();

        for (name, grad) in gradients {
            let kernel = build(grad.size(), device.props()).map_err(|e| format!("{e:?}"))?.compile(device.clone())?;

            sums.insert(name.clone(), Buffer::zeroed(device, DType::F32, grad.size())?);
            kernels.insert(name.clone(), kernel);
        }

        Ok(Self { sums, kernels, zero, one })
    }

    /// Adds `gradients` to the running sum.
    pub fn accumulate<'a>(
        &'a self,
        stream: &Arc<Stream<G>>,
        gradients: &TensorMap<G>,
        first: bool,
    ) -> Result<OptimiserUpdateSync<'a, G>, G::Error> {
        let keep = if first { &self.zero } else { &self.one };

        let mut sync = OptimiserUpdateSync::default();

        for (name, kernel) in &self.kernels {
            let inputs = vec![keep.clone(), gradients.get(name).unwrap().clone()];
            let outputs = vec![self.sums.get(name).unwrap().clone()];
            sync.push_kernel(kernel.execute(stream.clone(), inputs, outputs)?);
        }

        Ok(sync)
    }

    pub fn sums(&self) -> &TensorMap<G> {
        &self.sums
    }
}
