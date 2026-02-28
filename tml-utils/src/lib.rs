#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub type Float = f64;

#[doc(hidden)]
pub struct Assert<const CHECK: bool>;

#[doc(hidden)]
pub trait IsTrue {}

impl IsTrue for Assert<true> {}

#[macro_use]
mod tensor;

pub mod conv;
pub mod data;

pub use autodiff::{Gradients, Tape, Var};
pub use data::Sample;
pub use network::{DenseLayer, Flatten, Layer, ReLU, Sigmoid, TrainConfig, mse_loss};
pub use tensor::{Tensor, TensorView, TensorViewMut};

// helper stuff for proc macro
pub mod network;

// exposes `graph!` decl macro
pub mod autodiff;
