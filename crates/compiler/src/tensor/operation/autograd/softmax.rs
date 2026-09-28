use crate::tensor::{
    DType, IRTrace, Size, TNode, TType,
    operation::autograd::{CustomAutograd, CustomAutogradOp},
};

#[derive(Debug, PartialEq)]
pub struct SoftmaxCrossEntropyLoss {
    pub batch_size: Size,
    pub axis_size: usize,
}

impl SoftmaxCrossEntropyLoss {
    pub fn ttype(&self) -> TType {
        TType::new(self.batch_size * self.axis_size, DType::F32)
    }
}

impl CustomAutograd for SoftmaxCrossEntropyLoss {
    fn opname(&self) -> String {
        "softmax-cross-entropy-loss".into()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.ttype(), self.ttype()]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input, target] = inputs[..] else { return Err("Invalid number of inputs!".into()) };

        // log-softmax computed directly, since log(softmax(x)) produces -inf
        // (and then 0 * -inf = NaN) when a softmax entry underflows to zero
        let batch_size = self.batch_size;
        let inner = Size::from(self.axis_size);
        let max = input
            .reduce_max([batch_size, inner], 1)?
            .broadcast([batch_size, 1.into()], 1, self.axis_size)?;
        let shifted = (input - max)?;
        let log_denom = shifted
            .exp()?
            .reduce_sum([batch_size, inner], 1)?
            .log()?
            .broadcast([batch_size, 1.into()], 1, self.axis_size)?;
        let log_softmax = (shifted - log_denom)?;

        (-(target * log_softmax)?).map(|x| vec![x])
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input, target] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        let [grad] = output_grads[..] else { return Err("Invalid number of output grads!".into()) };
        let igrad = ((input.softmax(self.axis_size)? - target)? * grad)?;
        Ok(vec![igrad, target.zeros_like()])
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast() { self == other } else { false }
    }
}
