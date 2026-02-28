#![allow(incomplete_features)]
#![allow(unused)]
#![feature(generic_const_exprs)]

use tml::conv::Conv;
use tml::{Tensor, tensor};

fn main() {
    let mut tn = tensor!(2, 3);

    // println!("{}", type_of(&tn));
    // println!("{:?}", tn.get(0));

    println!("{}", tn.at([1, 1]));

    tn.set([1, 1], 2.);

    println!("{}", tn.at([1, 2]));

    #[rustfmt::skip]
    let mut c = Conv::<
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        0
    >::init();

    let mut avg_out_space = c.create_output_space();
    let mut cur_out_space = c.create_output_space();

    let n = 10000;

    for _ in 0..n {
        // re-randomize conv
        c = Conv::init();

        let input = c.input_from_data([1.; 8]);

        // dbg!(&c);
        // dbg!(&input);

        c.forward(&input, &mut cur_out_space);

        dbg!(&cur_out_space);

        // add assign requires the first variable to be `&mut`, not an owned value (which is what we need/like; see `tensor.rs` comment)
        avg_out_space += &cur_out_space;
    }

    dbg!(avg_out_space / n as f64);
    // fun fact: appears to converge to [[4.], [4.]]
}

// OUTPUT:

// tml::tensor::Tensor<12816, 15, [[[[[[[[f64; 4]; 3]; 2]; 1]; 1]; 89]; 3]; 2]>
// [tml/examples/conv_op.rs:25:5] &c = Conv {
//     data: [
//         Filter(
//             Tensor {
//                 data: [
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                 ],
//                 _shape_marker: PhantomData<[[[f64; 2]; 2]; 2]>,
//             },
//         ),
//         Filter(
//             Tensor {
//                 data: [
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                 ],
//                 _shape_marker: PhantomData<[[[f64; 2]; 2]; 2]>,
//             },
//         ),
//     ],
// }

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
