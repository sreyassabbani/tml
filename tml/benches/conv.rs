#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;
use tml::Float;
use tml::conv::{Conv, ConvOps};

fn fill_input<C: ConvOps>() -> C::InputArray {
    C::input_from_fn(|i| (i as Float) * 0.001)
}

fn bench_conv_case<C>(c: &mut Criterion, name: &str)
where
    C: ConvOps,
{
    let conv = C::init();
    let input: C::InputArray = fill_input::<C>();
    let mut output: C::OutputArray = C::output_zeroed();
    let id = format!("conv_forward/{name}");

    c.bench_function(&id, |b| {
        b.iter(|| {
            conv.forward_flat(black_box(&input), black_box(&mut output));
            black_box(&output);
        });
    });
}

fn conv_forward(c: &mut Criterion) {
    bench_conv_case::<Conv<8, 8, 1, 3, 3, 4, 1, 1>>(c, "tiny");
    bench_conv_case::<Conv<32, 32, 3, 3, 3, 8, 1, 1>>(c, "small");
    bench_conv_case::<Conv<64, 64, 3, 5, 5, 16, 1, 2>>(c, "medium");
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(8))
        .sample_size(60)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = conv_forward
}
criterion_main!(benches);
