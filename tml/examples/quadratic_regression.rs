use tml::{Float, Sample, TrainConfig, network};

fn main() {
    // y = 0.5x^2 - 1.0x + 0.2
    let mut model = network! {
        input(1) -> dense(16) -> relu -> dense(1) -> output
    };

    let samples = (-40..=40)
        .map(|i| {
            let x = i as Float / 10.0;
            let y = 0.5 * x * x - x + 0.2;
            Sample::new([x], [y])
        })
        .collect::<Vec<_>>();

    let config = TrainConfig {
        lr: 0.01,
        epochs: 2500,
    };
    let loss = model.fit_with(&samples, config);
    println!("final loss: {loss}");

    for x in [-2.5, -1.0, 0.0, 1.2, 2.5] {
        let target = 0.5 * x * x - x + 0.2;
        let pred = model.inference(&[x]);
        println!("x = {x:>5.2} -> y = {:.4} (target {:.4})", pred[0], target);
    }
}
