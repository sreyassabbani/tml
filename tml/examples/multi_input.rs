use tml::{Tape, graph};

use std::any::type_name_of_val;

fn main() {
    // Test single input graph (backward compatibility)
    let single_graph = graph! {
        input -> Sin -> Cos -> Pow(2) -> output
    };

    let (result, derivative) = single_graph.compute(1.0);
    println!(
        "Single input - f(1.0) = {:.6}, f'(1.0) = {:.6}",
        result, derivative
    );

    println!("{}", type_name_of_val(&single_graph));

    // Test multi-input graph with type-level arity
    let multi_graph = graph! {
        inputs: [x, z]
        x -> Pow(2) -> @x_sq
        z -> Cos -> @z_cos
        (@x_sq, @z_cos) -> Add -> @result
        output @result
    };

    println!("{}", type_name_of_val(&multi_graph));

    let mut tape = multi_graph.tape();
    let (result, grads) = multi_graph.compute_single_with_tape(&[2.0, 1.0], &mut tape);
    println!(
        "Multi input - f(2.0, 1.0) = {:.6}, grad = {:?}",
        result, grads
    );

    let mut tape = Tape::new();
    let x = tape.input("x", 2.0);
    let z = tape.input("z", 1.0);
    let res = (x.powi(2) + z.cos()).sin();
    let grads = tape.gradients(&res);
    println!(
        "Tape input - f(2.0, 1.0) = {:.6}, grads = {:?}",
        grads.value, grads.grads
    );

    // // Test mixed graph with type-level arity
    // let mut mixed_graph = graph! {
    //     inputs: [x, y]
    //     x -> pow(2) -> sin -> @temp1
    //     y -> cos -> scale(2.0) -> @temp2
    //     (@temp1, @temp2) -> mul -> @result
    //     output @result
    // };

    // let results = mixed_graph.compute(&[1.0, 0.5]);
    // if let Some((result, derivative)) = results.first() {
    //     println!(
    //         "Mixed graph - f(1.0, 0.5) = {:.6}, f'(1.0, 0.5) = {:.6}",
    //         result, derivative
    //     );
    // }

    // Test that type-level arity is enforced at compile time
    // This should cause a compilation error if we try to use wrong arity:
    // let mut invalid_graph = graph! {
    //     inputs: [x, y]
    //     x -> add -> @result  // This should fail - add needs 2 inputs
    //     output @result
    // };
}
