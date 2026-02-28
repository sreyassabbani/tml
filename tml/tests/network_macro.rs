#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::{Float, Sample, TrainConfig, network};

#[test]
fn type_contains_expected_layers() {
    let model = network! {
        input(4) -> dense(3) -> relu -> dense(2) -> output
    };
    let ty = std::any::type_name_of_val(&model);
    assert!(ty.contains("DenseLayer<4, 3>"), "type = {ty}");
    assert!(ty.contains("ReLU<3>"), "type = {ty}");
    assert!(ty.contains("DenseLayer<3, 2>"), "type = {ty}");
}

#[test]
fn zero_epochs_returns_zero_loss() {
    let mut model = network! {
        input(1) -> dense(1) -> output
    };
    let samples = vec![Sample::new([1.0], [5.0]), Sample::new([2.0], [8.0])];
    let loss = model.fit_with(&samples, TrainConfig { lr: 0.1, epochs: 0 });
    assert_eq!(loss, 0.0);
}

#[test]
fn conv_pipeline_infers_consistent_shape() {
    let model = network! {
        input(1, 4, 4) -> conv(2, 3, 1, 0) -> relu -> flatten -> dense(2) -> output
    };
    let out = model.inference(&[0.0 as Float; 16]);
    assert_eq!(out.len(), 2);
}
