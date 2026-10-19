use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    tensor::{
        DType, DValue, IRTrace, TType, TValue, TensorIR,
        operation::{CABinary, Matmul, MatrixLayout, PadAcrossDimension},
    },
};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    function::Function,
    runtime::{Device, Gpu, Stream},
};

use crate::model::TensorMap;

use super::TrainingError;

/// Computes grad norm.
pub struct GradNorms<G: Gpu> {
    func: Function<G>,
    inputs: BTreeMap<String, NodeId>,
    output: NodeId,
    norms: Arc<Buffer<G>>,
}

pub struct PendingGradNorms<'a, G: Gpu> {
    sync: SyncOnValue<G, &'a Function<G>>,
    norms: &'a GradNorms<G>,
}

fn build(gradients: &TensorMap<impl Gpu>) -> Result<(TensorIR, BTreeMap<String, NodeId>, NodeId), IRTrace> {
    let mut ir = TensorIR::default();
    let mut inputs = BTreeMap::new();
    let mut sqnorms = Vec::new();

    for (name, grad) in gradients {
        let size = grad.size();
        let input = ir.add_input(TType::new(size, DType::F32));

        // dot product g·g as 1×N @ N×1.
        let lhs = MatrixLayout { rows: 1.into(), cols: size.into(), col_mjr: true };
        let rhs = MatrixLayout { rows: size.into(), cols: 1.into(), col_mjr: true };
        sqnorms.push(ir.add_op([input, input], Matmul::new(DType::F32, 1, lhs, rhs))?[0]);

        inputs.insert(name.clone(), input);
    }

    // cat squared norms into a single vector
    // [a, b, c] = [a, 0, 0]
    //           + [0, b, 0]
    //           + [0, 0, c]
    let n = sqnorms.len();
    let mut cat = ir.add_scalar(DValue::zero(DType::F32), n);
    for (i, sqnorm) in sqnorms.into_iter().enumerate() {
        let pad = PadAcrossDimension::new([1], 0, i, n - 1 - i, DValue::zero(DType::F32));
        let term = ir.add_op([sqnorm], pad)?[0];
        cat = ir.add_binary(cat, term, CABinary::Add)?;
    }

    ir.register_output(cat);
    ir.optimise()?;

    Ok((ir, inputs, cat))
}

impl<G: Gpu> GradNorms<G> {
    pub fn new(device: &Arc<Device<G>>, gradients: &TensorMap<G>) -> Result<Self, TrainingError<G>> {
        let (ir, inputs, output) = build(gradients).map_err(TrainingError::CompilingBackwards)?;

        let mut func = Function::new(device.clone(), ir).map_err(TrainingError::CompilingBackwards)?;
        func.prealloc().map_err(TrainingError::Unexpected)?;

        let norms = Buffer::zeroed(device, DType::F32, inputs.len()).map_err(TrainingError::Unexpected)?;

        Ok(Self { func, inputs, output, norms })
    }

    pub fn execute(
        &self,
        stream: Arc<Stream<G>>,
        gradients: &TensorMap<G>,
    ) -> Result<PendingGradNorms<'_, G>, G::Error> {
        let mut map: BTreeMap<_, _> =
            self.inputs.iter().map(|(name, &id)| (id, gradients.get(name).unwrap().clone())).collect();
        map.insert(self.output, self.norms.clone());

        Ok(PendingGradNorms { sync: self.func.execute(stream, &map)?, norms: self })
    }
}

impl<G: Gpu> PendingGradNorms<'_, G> {
    pub fn read(self, stream: &Arc<Stream<G>>, scale: usize) -> Result<BTreeMap<String, f32>, G::Error> {
        self.sync.value()?;

        let TValue::F32(sqnorms) = self.norms.norms.to_host_async(stream)?.value()? else { panic!() };

        let norms = sqnorms.iter().map(|sq| sq.sqrt() / scale as f32);
        Ok(self.norms.inputs.keys().cloned().zip(norms).collect())
    }
}
