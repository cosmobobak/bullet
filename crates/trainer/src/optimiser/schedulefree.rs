use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use bullet_compiler::tensor::{DType, DValue, IRTrace, TType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    kernel::{CompiledKernel, KernelSrc},
    runtime::{Device, DeviceProps, Dialect, Dim3, Gpu, Stream},
};

use crate::{
    model::{ModelDefinition, ModelWeights},
    optimiser::{Optimiser, OptimiserUpdateResult, OptimiserUpdateSync},
};

use super::{OptimiserState, utils};

#[derive(Clone, Copy, Debug)]
pub struct ScheduleFreeAdamWParams {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for ScheduleFreeAdamWParams {
    fn default() -> Self {
        Self { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 }
    }
}

const OP_CUDA: &str = "\
__device__ __forceinline__ void sfadamOp(
    const float grad,
    const float rate,
    const float c,
    const int first,
    float* p,
    float* z,
    float* v
) {
    const float y = p[0];
    const float z_old = first ? y : z[0];

    v[0] = static_cast<float>(BETA2) * v[0] + (1.0F - static_cast<float>(BETA2)) * grad * grad;

    const float normed = grad / (sqrtf(v[0]) + static_cast<float>(EPSILON));
    const float u = normed + static_cast<float>(DECAY) * y;

    z[0] = z_old - rate * u;

    const float p_new = (1.0F - c) * y + c * z_old - rate * u * (1.0F - static_cast<float>(BETA1) * (1.0F - c));
    p[0] = min(max(p_new, static_cast<float>(WMIN)), static_cast<float>(WMAX));
}";

const DECL_CUDA: &str = "
extern \"C\" __global__ void sfadamw(
    const float* adj_ptr,
    const float* rate_ptr,
    const float* c_ptr,
    const int* first_ptr,
    const float* gradients,
    float* network,
    float* fast,
    float* velocity
)";

impl ScheduleFreeAdamWParams {
    pub fn build(&self, size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
        let (op_src, decl) = match props.dialect() {
            Dialect::CudaHip => (OP_CUDA, DECL_CUDA),
            Dialect::Msl => todo!(),
        };

        let op = op_src
            .replace("DECAY", &format!("{:.E}", self.decay))
            .replace("BETA1", &format!("{:.E}", self.beta1))
            .replace("BETA2", &format!("{:.E}", self.beta2))
            .replace("WMIN", &format!("{:.E}", self.min_weight))
            .replace("WMAX", &format!("{:.E}", self.max_weight))
            .replace("EPSILON", "0.00000001F");

        let body = match props.dialect() {
            Dialect::CudaHip => {
                if size.is_multiple_of(4) {
                    format!(
                        "
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < {})
                {{
                    const float adj = adj_ptr[0];
                    const float rate = rate_ptr[0];
                    const float c = c_ptr[0];
                    const int first = first_ptr[0];
                    float4 p = ((float4 *)network)[tid];
                    float4 z = ((float4 *)fast)[tid];
                    float4 v = ((float4 *)velocity)[tid];
                    const float4 g = ((const float4 *)gradients)[tid];

                    sfadamOp(adj * g.x, rate, c, first, &p.x, &z.x, &v.x);
                    sfadamOp(adj * g.y, rate, c, first, &p.y, &z.y, &v.y);
                    sfadamOp(adj * g.z, rate, c, first, &p.z, &z.z, &v.z);
                    sfadamOp(adj * g.w, rate, c, first, &p.w, &z.w, &v.w);

                    ((float4 *)network)[tid] = p;
                    ((float4 *)fast)[tid] = z;
                    ((float4 *)velocity)[tid] = v;
                }}",
                        size / 4,
                    )
                } else {
                    format!(
                        "
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < {size})
                {{
                    const float adj = adj_ptr[0];
                    const float rate = rate_ptr[0];
                    const float c = c_ptr[0];
                    const int first = first_ptr[0];
                    float p = network[tid];
                    float z = fast[tid];
                    float v = velocity[tid];
                    const float g = gradients[tid];

                    sfadamOp(adj * g, rate, c, first, &p, &z, &v);

                    network[tid] = p;
                    fast[tid] = z;
                    velocity[tid] = v;
                }}"
                    )
                }
            }
            Dialect::Msl => todo!(),
        };

        let ty = TType::new(size, DType::F32);
        let sty = TType::new(1, DType::F32);

        let total_threads = if size.is_multiple_of(4) { size / 4 } else { size };
        let src = unsafe {
            KernelSrc::new(
                vec![sty, sty, sty, TType::new(1, DType::I32), ty],
                vec![ty; 3],
                "sfadamw".to_string(),
                format!("{op}{decl}{{{body}}}"),
                vec![(0, true), (1, true), (2, true), (3, true), (4, true), (0, false), (1, false), (2, false)],
                BTreeSet::new(),
                Dim3 { x: total_threads.div_ceil(256) as u32, y: 1, z: 1 },
                256,
                0,
            )
        };

        Ok(src)
    }
}

/// Kernel converting the network buffer in place via `p <- p + w * (z - p)`.
/// With `w = 1 - 1/beta1` this maps `y -> x`, with `w = 1 - beta1` it maps
/// `x -> y`.
fn build_convert_op(size: usize, props: &DeviceProps) -> Result<KernelSrc, IRTrace> {
    let decl = match props.dialect() {
        Dialect::CudaHip => {
            "
extern \"C\" __global__ void sfconvert(
    const float* w_ptr,
    const float* fast,
    float* network
)"
        }
        Dialect::Msl => todo!(),
    };

    let body = match props.dialect() {
        Dialect::CudaHip => format!(
            "
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < {size})
                {{
                    const float w = w_ptr[0];
                    const float p = network[tid];
                    const float z = fast[tid];
                    network[tid] = p + w * (z - p);
                }}"
        ),
        Dialect::Msl => todo!(),
    };

    let ty = TType::new(size, DType::F32);
    let sty = TType::new(1, DType::F32);

    let src = unsafe {
        KernelSrc::new(
            vec![sty, ty],
            vec![ty],
            "sfconvert".to_string(),
            format!("{decl}{{{body}}}"),
            vec![(0, true), (1, true), (0, false)],
            BTreeSet::new(),
            Dim3 { x: size.div_ceil(256) as u32, y: 1, z: 1 },
            256,
            0,
        )
    };

    Ok(src)
}

pub struct ScheduleFreeAdamW<G: Gpu> {
    /// `z` in the schedule-free update rules.
    fast: Arc<Buffer<G>>,
    velocity: Arc<Buffer<G>>,
    op: CompiledKernel<G>,
    convert_op: CompiledKernel<G>,
    params: ScheduleFreeAdamWParams,
    step: usize,
    /// whether the network buffer currently holds `x` rather than `y`.
    eval: bool,
    c: Arc<Buffer<G>>,
    first: Arc<Buffer<G>>,
    conv_w: Arc<Buffer<G>>,
    cpu_c: TValue,
    cpu_first: TValue,
    cpu_conv_w: TValue,
}

impl<G: Gpu> ScheduleFreeAdamW<G> {
    pub fn new(
        definition: ModelDefinition,
        weights: ModelWeights,
        device: Arc<Device<G>>,
        params: ScheduleFreeAdamWParams,
    ) -> Result<Optimiser<G, Self>, G::Error> {
        Optimiser::new(definition, weights, device, params)
    }

    fn convert<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        to_eval: bool,
    ) -> OptimiserUpdateResult<'a, G> {
        if self.eval == to_eval || self.step == 0 {
            return Ok(OptimiserUpdateSync::default());
        }

        self.eval = to_eval;

        let beta1 = self.params.beta1;
        let w = if to_eval { 1.0 - 1.0 / beta1 } else { 1.0 - beta1 };
        self.cpu_conv_w.write(0, DValue::F32(w));

        let mut sync = OptimiserUpdateSync::default();
        sync.push_copy(self.conv_w.copy_from_host_async(stream, &self.cpu_conv_w)?);
        sync.push_kernel(self.convert_op.execute(
            stream.clone(),
            vec![self.conv_w.clone(), self.fast.clone()],
            vec![weights],
        )?);

        Ok(sync)
    }
}

impl<G: Gpu> OptimiserState<G> for ScheduleFreeAdamW<G> {
    type Params = ScheduleFreeAdamWParams;

    fn new(device: &Arc<Device<G>>, size: usize, default_params: Self::Params) -> Result<Self, G::Error> {
        if default_params.max_weight < default_params.min_weight {
            return Err(
                format!("Invalid clipping: {} >= {}", default_params.min_weight, default_params.max_weight).into()
            );
        }

        let op = default_params.build(size, device.props()).unwrap().compile(device.clone())?;
        let convert_op = build_convert_op(size, device.props()).unwrap().compile(device.clone())?;

        Ok(Self {
            fast: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            velocity: Buffer::from_host(device, &TValue::zeros(DType::F32, size))?,
            op,
            convert_op,
            params: default_params,
            step: 0,
            eval: false,
            c: Buffer::from_host(device, &TValue::zeros(DType::F32, 1))?,
            first: Buffer::from_host(device, &TValue::zeros(DType::I32, 1))?,
            conv_w: Buffer::from_host(device, &TValue::zeros(DType::F32, 1))?,
            cpu_c: TValue::F32(vec![0.0]),
            cpu_first: TValue::I32(vec![0]),
            cpu_conv_w: TValue::F32(vec![0.0]),
        })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        // so we can get c = 1/(t+1).
        let step = self.step;
        let c = 1.0 / (step as f32 + 1.0);
        let first = i32::from(step == 0);

        self.cpu_c.write(0, DValue::F32(c));
        self.cpu_first.write(0, DValue::I32(first));

        self.step = step + 1;

        let mut sync = OptimiserUpdateSync::default();

        sync.push_copy(self.c.copy_from_host_async(stream, &self.cpu_c)?);
        sync.push_copy(self.first.copy_from_host_async(stream, &self.cpu_first)?);

        sync.push_kernel(self.op.execute(
            stream.clone(),
            vec![gradient_factor, learning_rate, self.c.clone(), self.first.clone(), grads],
            vec![weights, self.fast.clone(), self.velocity.clone()],
        )?);

        Ok(sync)
    }

    fn convert_to_eval<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        self.convert(stream, weights, true)
    }

    fn convert_to_train<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        self.convert(stream, weights, false)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        let size = self.fast.size();
        self.fast.copy_from_host(&TValue::zeros(DType::F32, size))?;
        self.velocity.copy_from_host(&TValue::zeros(DType::F32, size))?;
        self.step = 0;
        self.eval = false;
        Ok(())
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let fast: Vec<_> = map.iter().map(|(id, single)| (id, &single.fast)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        utils::write_weights_to_file::<G>(&fast, &format!("{path}/z.bin"))?;
        utils::write_weights_to_file::<G>(&velocity, &format!("{path}/velocity.bin"))?;

        let mut file = File::create(format!("{path}/step.txt")).unwrap();
        for (id, single) in map.iter() {
            writeln!(file, "{id},{}", single.step).unwrap();
        }

        Ok(())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let mut fast = utils::load_weights_from_file(&format!("{path}/z.bin"));
        let mut velocity = utils::load_weights_from_file(&format!("{path}/velocity.bin"));

        let file = File::open(format!("{path}/step.txt")).unwrap();
        let mut steps = BufReader::new(file)
            .lines()
            .map(|s| {
                let s = s.unwrap();
                let mut split = s.split(',');
                let id = split.next().unwrap();
                (id.to_string(), split.next().unwrap().parse().unwrap())
            })
            .collect::<Vec<(String, usize)>>();

        fast.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());
        steps.sort_by_key(|(id, _)| id.clone());

        for (((id1, z), (id2, vel)), (id3, step)) in fast.into_iter().zip(velocity).zip(steps) {
            assert_eq!(id1, id2);
            assert_eq!(id1, id3);

            let single = map.get_mut(&id1).unwrap();
            single.fast.copy_from_host(&TValue::F32(z))?;
            single.velocity.copy_from_host(&TValue::F32(vel))?;
            single.step = step;
            single.eval = false;
        }

        Ok(())
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.params = params;
        let size = self.fast.size();
        let device = self.fast.device();
        self.op = params.build(size, device.props()).unwrap().compile(device)?;
        Ok(())
    }
}
