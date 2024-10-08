use crate::tensor::{DenseMatrix, Shape, Tensor};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 1 {
        let input = inputs[0];
        if input.rows() % 2 != 0 {
            Err(String::from("Input size must be even!"))
        } else {
            Ok(Shape::new(input.rows() / 2, input.cols()))
        }
    } else {
        Err(format!("Invalid number of inputs in pairwise! Expected 1, got {}", inputs.len()))
    }
}

pub fn forward(inputs: &[&Tensor], output: &mut Tensor, pc: bool) {
    DenseMatrix::pairwise(inputs[0].values.dense(), output.values.dense_mut(), pc);
}

pub fn backprop(output: &Tensor, inputs: &mut [&mut Tensor], pc: bool) {
    let input = inputs[0].values.dense();
    let output_grad = output.gradients.as_ref().expect("Must exist!");
    if let Some(input_grad) = inputs[0].gradients.as_mut() {
        DenseMatrix::backprop_pairwise(input, output_grad, input_grad, pc);
    }
}
