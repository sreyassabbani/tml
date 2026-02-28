use crate::network::Layer;
use crate::{tensor::Tensor, Assert, Float, IsTrue};
use std::{array, marker::PhantomData};

#[doc(hidden)]
pub const fn conv_out_dim(input: usize, pad: usize, kernel: usize, stride: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    let padded = input + 2 * pad;
    if padded < kernel {
        return 0;
    }
    let numer = padded - kernel;
    if numer % stride != 0 {
        return 0;
    }
    numer / stride + 1
}

#[derive(Debug, Clone)]
pub struct Filter<const H: usize, const W: usize, const D: usize>
where
    [(); H * W * D]:,
{
    weights: Tensor<{ H * W * D }, 3, shape_ty!(H, W, D)>,
    grads: Box<[Float; H * W * D]>,
}

impl<const H: usize, const W: usize, const D: usize> Default for Filter<H, W, D>
where
    Tensor<{ H * W * D }, 3, shape_ty!(H, W, D)>: Sized,
    [(); H * W * D]:,
{
    fn default() -> Self {
        let mut arr = [0.0 as Float; H * W * D];
        for v in &mut arr {
            *v = rand::random::<Float>();
        }

        Self {
            weights: Tensor {
                data: Box::new(arr),
                _shape_marker: PhantomData,
            },
            grads: Box::new([0.0 as Float; H * W * D]),
        }
    }
}

impl<const H: usize, const W: usize, const D: usize> Filter<H, W, D>
where
    Tensor<{ H * W * D }, 3, shape_ty!(H, W, D)>: Sized,
    [(); H * W * D]:,
{
    fn weights(&self) -> &[Float] {
        self.weights.as_slice()
    }

    fn grads_mut(&mut self) -> &mut [Float] {
        &mut self.grads[..]
    }
}

/// A convolutional layer
///
/// `FH` - filter/kernel height
/// `FW` - filter/kernel width
/// `IC` - number of input channels
/// `OC` - number of output channels (equivalently, number of kernels/filters)
/// `S` - stride
/// `P` - padding
#[derive(Debug)]
pub struct Conv<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const FH: usize,
    const FW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> where
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
{
    filters: [Filter<FH, FW, IC>; OC],
    biases: Box<[Float; OC]>,
    bias_grads: Box<[Float; OC]>,
}
impl<
        const IW: usize,
        const IH: usize,
        const IC: usize,
        const FH: usize,
        const FW: usize,
        const OC: usize,
        const S: usize,
        const P: usize,
    > Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    Assert<{ conv_out_dim(IH, P, FH, S) > 0 }>: IsTrue,
    Assert<{ conv_out_dim(IW, P, FW, S) > 0 }>: IsTrue,
{
    pub fn init() -> Self {
        Conv {
            filters: array::from_fn(|_| Filter::default()),
            biases: Box::new([0.0 as Float; OC]),
            bias_grads: Box::new([0.0 as Float; OC]),
        }
    }

    pub fn create_output_space(&self) -> <Self as ConvIO>::Output {
        Tensor::<
            { OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S) },
            3,
            shape_ty!(OC, conv_out_dim(IH, P, FH, S), conv_out_dim(IW, P, FW, S)),
        > {
            data: Box::new(
                [0.0 as Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
            ),
            _shape_marker: PhantomData,
        }
    }

    pub fn input_from_data(&self, data: [Float; IC * IH * IW]) -> <Self as ConvIO>::Input {
        Tensor::<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)> {
            data: Box::new(data),
            _shape_marker: PhantomData,
        }
    }

    pub fn forward(
        &self,
        input: &Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>,
        output: &mut Tensor<
            { OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S) },
            3,
            shape_ty!(OC, conv_out_dim(IH, P, FH, S), conv_out_dim(IW, P, FW, S)),
        >,
    ) {
        let input_arr: &[Float; IC * IH * IW] = input.as_slice().try_into().expect("bad input");
        let output_arr: &mut [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)] =
            output.as_mut_slice().try_into().expect("bad output");
        self.forward_flat(input_arr, output_arr);
    }

    pub fn forward_flat(
        &self,
        input: &[Float; IC * IH * IW],
        output: &mut [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
    ) {
        let out_h = conv_out_dim(IH, P, FH, S);
        let out_w = conv_out_dim(IW, P, FW, S);

        for oc in 0..OC {
            let filter_data = self.filters[oc].weights();

            for y in 0..out_h {
                for x in 0..out_w {
                    let mut sum = self.biases[oc];

                    for ky in 0..FH {
                        for kx in 0..FW {
                            for ic in 0..IC {
                                let in_y = y * S + ky;
                                let in_x = x * S + kx;
                                let in_y = in_y as isize - P as isize;
                                let in_x = in_x as isize - P as isize;

                                if in_y >= 0
                                    && in_y < IH as isize
                                    && in_x >= 0
                                    && in_x < IW as isize
                                {
                                    let in_y = in_y as usize;
                                    let in_x = in_x as usize;
                                    let input_idx = ic * IH * IW + in_y * IW + in_x;
                                    let filter_idx = (ky * FW + kx) * IC + ic;
                                    sum += filter_data[filter_idx] * input[input_idx];
                                }
                            }
                        }
                    }

                    let output_idx = oc * out_h * out_w + y * out_w + x;
                    output[output_idx] = sum;
                }
            }
        }
    }

    pub fn backward_flat(
        &mut self,
        input: &[Float; IC * IH * IW],
        output_grad: &[Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
        input_grad: &mut [Float; IC * IH * IW],
        lr: Float,
    ) {
        let out_h = conv_out_dim(IH, P, FH, S);
        let out_w = conv_out_dim(IW, P, FW, S);

        input_grad.fill(0.0);
        self.bias_grads.fill(0.0);
        for filter in &mut self.filters {
            filter.grads_mut().fill(0.0);
        }

        for oc in 0..OC {
            let Filter { weights, grads } = &mut self.filters[oc];
            let filter_weights = weights.as_slice();
            let filter_grads = &mut grads[..];

            for y in 0..out_h {
                for x in 0..out_w {
                    let output_idx = oc * out_h * out_w + y * out_w + x;
                    let grad = output_grad[output_idx];
                    self.bias_grads[oc] += grad;

                    for ky in 0..FH {
                        for kx in 0..FW {
                            for ic in 0..IC {
                                let in_y = y * S + ky;
                                let in_x = x * S + kx;
                                let in_y = in_y as isize - P as isize;
                                let in_x = in_x as isize - P as isize;

                                if in_y >= 0
                                    && in_y < IH as isize
                                    && in_x >= 0
                                    && in_x < IW as isize
                                {
                                    let in_y = in_y as usize;
                                    let in_x = in_x as usize;
                                    let input_idx = ic * IH * IW + in_y * IW + in_x;
                                    let filter_idx = (ky * FW + kx) * IC + ic;

                                    filter_grads[filter_idx] += grad * input[input_idx];
                                    input_grad[input_idx] += grad * filter_weights[filter_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        for oc in 0..OC {
            let Filter { weights, grads } = &mut self.filters[oc];
            let weights = weights.as_mut_slice();
            for i in 0..weights.len() {
                weights[i] -= lr * grads[i];
                grads[i] = 0.0;
            }

            self.biases[oc] -= lr * self.bias_grads[oc];
            self.bias_grads[oc] = 0.0;
        }
    }
}

impl<
        const IW: usize,
        const IH: usize,
        const IC: usize,
        const FH: usize,
        const FW: usize,
        const OC: usize,
        const S: usize,
        const P: usize,
    > Layer<{ IC * IH * IW }, { OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S) }>
    for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    Assert<{ conv_out_dim(IH, P, FH, S) > 0 }>: IsTrue,
    Assert<{ conv_out_dim(IW, P, FW, S) > 0 }>: IsTrue,
{
    fn forward(
        &self,
        input: &[Float; IC * IH * IW],
        output: &mut [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
    ) {
        self.forward_flat(input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; IC * IH * IW],
        _output: &[Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
        output_grad: &[Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
        input_grad: &mut [Float; IC * IH * IW],
        lr: Float,
    ) {
        self.backward_flat(input, output_grad, input_grad, lr);
    }
}

#[allow(dead_code)]
/// Type-level input/output tensor metadata for conv layers.
pub trait ConvIO {
    type Output;
    type Input;
    type OutputShape;
    type InputShape;
    type FilterShape;
    const N: usize;
}

impl<
        const IW: usize,
        const IH: usize,
        const IC: usize,
        const FH: usize,
        const FW: usize,
        const OC: usize,
        const S: usize,
        const P: usize,
    > ConvIO for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>: Sized,
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    Tensor<
        { OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S) },
        3,
        shape_ty!(OC, conv_out_dim(IH, P, FH, S), conv_out_dim(IW, P, FW, S)),
    >: Sized,
{
    const N: usize = IC * IH * IW;
    type Input = Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>;
    type Output = Tensor<
        { OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S) },
        3,
        Self::OutputShape,
    >;
    type InputShape = shape_ty!(IC, IH, IW);
    type OutputShape = shape_ty!(OC, conv_out_dim(IH, P, FH, S), conv_out_dim(IW, P, FW, S));
    type FilterShape = shape_ty!(FH, FW, IC);
}

#[allow(dead_code)]
/// Flat-array convenience trait for generic conv code.
pub trait ConvOps: ConvIO {
    type InputArray;
    type OutputArray;
    type FilterArray;

    const INPUT_SIZE: usize;
    const OUTPUT_SIZE: usize;
    const FILTER_SIZE: usize;

    fn init() -> Self;
    fn forward_flat(&self, input: &Self::InputArray, output: &mut Self::OutputArray);
    fn input_from_fn<F: FnMut(usize) -> Float>(f: F) -> Self::InputArray;
    fn output_zeroed() -> Self::OutputArray;
}

impl<
        const IW: usize,
        const IH: usize,
        const IC: usize,
        const FH: usize,
        const FW: usize,
        const OC: usize,
        const S: usize,
        const P: usize,
    > ConvOps for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
    [(); FH * FW * IC]:,
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    Assert<{ conv_out_dim(IH, P, FH, S) > 0 }>: IsTrue,
    Assert<{ conv_out_dim(IW, P, FW, S) > 0 }>: IsTrue,
{
    type InputArray = [Float; IC * IH * IW];
    type OutputArray = [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)];
    type FilterArray = [Float; FH * FW * IC];

    const INPUT_SIZE: usize = IC * IH * IW;
    const OUTPUT_SIZE: usize = OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S);
    const FILTER_SIZE: usize = FH * FW * IC;

    fn init() -> Self {
        Conv::<IW, IH, IC, FH, FW, OC, S, P>::init()
    }

    fn forward_flat(&self, input: &Self::InputArray, output: &mut Self::OutputArray) {
        Conv::<IW, IH, IC, FH, FW, OC, S, P>::forward_flat(self, input, output);
    }

    fn input_from_fn<F: FnMut(usize) -> Float>(mut f: F) -> Self::InputArray {
        array::from_fn(|i| f(i))
    }

    fn output_zeroed() -> Self::OutputArray {
        array::from_fn(|_| 0.0 as Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type ConvCase = Conv<3, 3, 1, 2, 2, 1, 1, 0>;
    const IN_SIZE: usize = 3 * 3;
    const OUT_SIZE: usize = 4;

    fn approx_eq(a: Float, b: Float, eps: Float) {
        let diff = (a - b).abs();
        assert!(diff <= eps, "expected {a} ~= {b} (diff={diff}, eps={eps})");
    }

    fn configured_conv() -> ConvCase {
        let mut conv = ConvCase::init();
        for (i, w) in conv.filters[0]
            .weights
            .as_mut_slice()
            .iter_mut()
            .enumerate()
        {
            *w = 0.1 * (i as Float + 1.0);
        }
        conv.biases[0] = 0.05;
        conv
    }

    fn objective(
        conv: &ConvCase,
        input: &[Float; IN_SIZE],
        output_grad: &[Float; OUT_SIZE],
    ) -> Float {
        let mut output = [0.0; OUT_SIZE];
        conv.forward_flat(input, &mut output);
        output
            .iter()
            .zip(output_grad.iter())
            .map(|(o, g)| o * g)
            .sum()
    }

    #[test]
    fn input_gradient_matches_finite_difference() {
        let mut conv = configured_conv();
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let output_grad = [0.3, -0.2, 0.1, 0.4];
        let mut input_grad = [0.0; IN_SIZE];

        conv.backward_flat(&input, &output_grad, &mut input_grad, 0.0);

        let eps = 1e-7;
        for i in 0..IN_SIZE {
            let mut plus = input;
            let mut minus = input;
            plus[i] += eps;
            minus[i] -= eps;
            let f_plus = objective(&conv, &plus, &output_grad);
            let f_minus = objective(&conv, &minus, &output_grad);
            let numeric = (f_plus - f_minus) / (2.0 * eps);
            approx_eq(input_grad[i], numeric, 1e-6);
        }
    }

    #[test]
    fn weight_update_matches_finite_difference_gradient() {
        let mut conv = configured_conv();
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let output_grad = [0.3, -0.2, 0.1, 0.4];
        let mut input_grad = [0.0; IN_SIZE];
        let weight_idx = 2;

        let eps = 1e-7;
        let mut conv_plus = configured_conv();
        conv_plus.filters[0].weights.as_mut_slice()[weight_idx] += eps;
        let mut conv_minus = configured_conv();
        conv_minus.filters[0].weights.as_mut_slice()[weight_idx] -= eps;
        let numeric = (objective(&conv_plus, &input, &output_grad)
            - objective(&conv_minus, &input, &output_grad))
            / (2.0 * eps);

        let lr = 1e-3;
        let before = conv.filters[0].weights.as_slice()[weight_idx];
        conv.backward_flat(&input, &output_grad, &mut input_grad, lr);
        let after = conv.filters[0].weights.as_slice()[weight_idx];
        let analytic = (before - after) / lr;

        approx_eq(analytic, numeric, 1e-6);
    }
}
