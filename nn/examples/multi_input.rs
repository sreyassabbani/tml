use nn::graph;

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
    let results = multi_graph.compute_with_tape(&[2.0, 1.0], &mut tape);
    if let Some((result, derivative)) = results.first() {
        println!(
            "Multi input - f(2.0, 1.0) = {:.6}, f'(2.0, 1.0) = {:.6}",
            result, derivative
        );
    }

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
