#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::{Float, Sample, TrainConfig, network};

const C: usize = 1;
const H: usize = 4;
const W: usize = 4;
const INPUT: usize = C * H * W;
const OUT: usize = 2;

fn main() {
    let mut model = network! {
        input(C, H, W) -> conv(2, 3, 1, 0) -> relu -> flatten -> dense(OUT) -> output
    };

    let samples: Vec<_> = (1..=3)
        .flat_map(|idx| [gen_vertical(idx), gen_horizontal(idx)])
        .map(Sample::from)
        .collect();

    let config = TrainConfig {
        lr: 0.05,
        epochs: 2000,
    };

    let loss = model.fit_with(&samples, config);
    println!("final loss: {loss}");

    for (idx, sample) in samples.iter().enumerate() {
        let output = model.inference(&sample.input);
        let pred = argmax(&output);
        println!("sample {idx}: pred={pred} logits={output:?}");
    }
}

fn argmax<const N: usize>(values: &[Float; N]) -> usize {
    let mut best = 0;
    let mut best_val = values[0];
    for (i, &value) in values.iter().enumerate().skip(1) {
        if value > best_val {
            best_val = value;
            best = i;
        }
    }
    best
}

fn gen_vertical(col: usize) -> ([Float; INPUT], [Float; OUT]) {
    (vertical_line(col), one_hot(0))
}

fn gen_horizontal(row: usize) -> ([Float; INPUT], [Float; OUT]) {
    (horizontal_line(row), one_hot(1))
}

fn one_hot(label: usize) -> [Float; OUT] {
    let mut out = [0.0; OUT];
    out[label] = 1.0;
    out
}

fn vertical_line(col: usize) -> [Float; INPUT] {
    let mut img = [0.0; INPUT];
    for y in 0..H {
        let idx = y * W + col;
        img[idx] = 1.0;
    }
    img
}

fn horizontal_line(row: usize) -> [Float; INPUT] {
    let mut img = [0.0; INPUT];
    let base = row * W;
    for x in 0..W {
        img[base + x] = 1.0;
    }
    img
}
