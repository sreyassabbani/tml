#![allow(internal_features)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(core_intrinsics)]
#![feature(generic_const_items)]
#![feature(specialization)]

#[macro_use]
mod tensor;

pub mod conv;

pub use layerable::{LayerKind, Layerable};
pub use tensor::Tensor;

// helper stuff for proc macro
pub mod network;

// exposes `graph!` decl macro
pub mod autodiff;

pub mod layerable;
