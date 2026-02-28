use crate::Float;

pub trait Layer<const IN: usize, const OUT: usize> {
    fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]);
    fn backward(
        &mut self,
        input: &[Float; IN],
        output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
        lr: Float,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub lr: Float,
    pub epochs: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            epochs: 1,
        }
    }
}

pub fn mse_loss<const N: usize>(
    output: &[Float; N],
    target: &[Float; N],
    grad: &mut [Float; N],
) -> Float {
    let scale = 2.0 / N as Float;
    let loss = output
        .iter()
        .zip(target.iter())
        .zip(grad.iter_mut())
        .map(|((&o, &t), g)| {
            let diff = o - t;
            *g = diff * scale;
            diff * diff
        })
        .sum::<Float>();
    loss / N as Float
}

// Define the DenseLayer struct with weights and biases
#[derive(Debug)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: Box<[[Float; IN]; OUT]>,
    biases: Box<[Float; OUT]>,
}

// Rectified Linear Unit
#[derive(Debug)]
pub struct ReLU<const N: usize>;

// Sigmoid
#[derive(Debug)]
pub struct Sigmoid<const N: usize>;

#[derive(Debug)]
pub struct Flatten<const N: usize>;

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn init() -> Self {
        let mut weights = [[0.0; IN]; OUT];
        let scale = (1.0 / IN as Float).sqrt();
        for row in &mut weights {
            for v in row {
                *v = (rand::random::<Float>() * 2.0 - 1.0) * scale;
            }
        }
        Self {
            weights: Box::new(weights),
            biases: Box::new([0.0; OUT]),
        }
    }

    pub fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]) {
        for (o, out) in output.iter_mut().enumerate() {
            let mut sum = self.biases[o];
            for (i, &inp) in input.iter().enumerate() {
                sum += self.weights[o][i] * inp;
            }
            *out = sum;
        }
    }

    pub fn backward(
        &mut self,
        input: &[Float; IN],
        _output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
        lr: Float,
    ) {
        input_grad.fill(0.0);

        for (o, &grad) in output_grad.iter().enumerate() {
            for (i, in_grad) in input_grad.iter_mut().enumerate() {
                *in_grad += self.weights[o][i] * grad;
            }
        }

        for (o, &grad) in output_grad.iter().enumerate() {
            self.biases[o] -= lr * grad;
            for (i, &inp) in input.iter().enumerate() {
                self.weights[o][i] -= lr * grad * inp;
            }
        }
    }
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> for DenseLayer<IN, OUT> {
    fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]) {
        DenseLayer::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; IN],
        output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
        lr: Float,
    ) {
        DenseLayer::backward(self, input, output, output_grad, input_grad, lr);
    }
}

impl<const N: usize> ReLU<N> {
    pub fn init() -> Self {
        ReLU
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        for i in 0..N {
            output[i] = input[i].max(0.0);
        }
    }

    pub fn backward(
        &self,
        input: &[Float; N],
        _output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        for i in 0..N {
            input_grad[i] = if input[i] > 0.0 { output_grad[i] } else { 0.0 };
        }
    }
}

impl<const N: usize> Sigmoid<N> {
    pub fn init() -> Self {
        Sigmoid
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        for i in 0..N {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
    }

    pub fn backward(
        &self,
        _input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        for i in 0..N {
            let y = output[i];
            input_grad[i] = output_grad[i] * y * (1.0 - y);
        }
    }
}

impl<const N: usize> Flatten<N> {
    pub fn init() -> Self {
        Flatten
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        output.copy_from_slice(input);
    }

    pub fn backward(
        &self,
        _input: &[Float; N],
        _output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        input_grad.copy_from_slice(output_grad);
    }
}

impl<const N: usize> Layer<N, N> for ReLU<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        ReLU::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
        _lr: Float,
    ) {
        ReLU::backward(self, input, output, output_grad, input_grad);
    }
}

impl<const N: usize> Layer<N, N> for Sigmoid<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        Sigmoid::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
        _lr: Float,
    ) {
        Sigmoid::backward(self, input, output, output_grad, input_grad);
    }
}

impl<const N: usize> Layer<N, N> for Flatten<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        Flatten::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
        _lr: Float,
    ) {
        Flatten::backward(self, input, output, output_grad, input_grad);
    }
}
