# Development

1. Clone the repo.

Notice that in order to run this, I have `rust-toolchain.toml` set to `toolchain.channel = "nightly"`. You may also opt to have control of every commands by selecting `cargo +nightly ...`.

> [!NOTE]
> Examples and downstream crates that use most of the const-generic API must enable:
>
> ```rs
> #![feature(generic_const_exprs)]
> #![allow(incomplete_features)]
> ```
>
> This is required for the compiler to evaluate the shape constraints.

> [!NOTE]
> For the development philosophy, goals, and design decisions for this library, see [DESIGN](DESIGN.md).

# Explanations

## The `network!` proc macro

This macro is the spotlight of this library. The goal of it is to generate a struct `Network<Layers>`. The generic parameter here is a tuple of every layer that is supplied.

For example, executing `cargo run --example linear-regression --features="unstable"`, you will see the following type debugged (here it is formatted prettier):

```rs
linear_regression::main::Network<(
  tml::network::DenseLayer<784, 128>,
  tml::network::ReLU<128>,
  tml::network::DenseLayer<128, 64>,
  tml::network::Sigmoid<64>,
  tml::network::DenseLayer<64, 10>
)>
```

While parsing macro contents, `proc_macro2::TokenStream`s from `quote!` are declaratively collected into special variables in _three_ main stages:

In the first stage, an instance of `parsing::NetworkDef` is formed. This is the first step done through `syn::parse_macro_input!`. These are

- `input_shape: (N) or (C, H, W)`
- `layers: Vec<Layer>`

Then, in the second stage, the following constants are generated, all extracted from `layers` and `input_size`

- `layer_types`
- `forward_calls`
- `layer_inits`
- other buffer setup (`max_size` from `layers` `use_buf_a`, `final_buffer`)

<details>
  <summary>The <code>parsing::Layer</code> type</summary>
  
  An enum defined as
  
  ```rs
    pub enum Layer {
      Conv {
          /// Number of output channels/features in the output. Alternatively, this may be interpreted as the number of filters in the convolutional layer.
          out_channels: usize,
          kernel_h: usize,
          kernel_w: usize,
          stride: usize,
          padding: usize,
      },
      Dense(usize),
      ReLU,
      Sigmoid,
      Flatten,
  }
  ```
</details>

??? There is a lot of bypassing that is done especially around the Rust orphan rule by defining structs temporarily during the expansion of the macro.

# Development of the `Tensor`

Goals, like [always](DESIGN.md), are to utilize as many zero-cost abstractions as possible, parse and not validate, etc.

- I would like to keep the dimensions of the tensor part of the type information while also being extremely general. That is, I don't want to separately define (or even macro) `Tensor2x3` or `Tensor2x4x2`, etc.
- Moreover, I would like a convenient `reshape()` functionality. As we are going to store the tensor data contiguously, I would like to be able to create separate "indices/views" onto the tensor.

Looking at both requirements at the same time, we could do either:

1. `Tensor<(1, 2, 3)>`
   1a.`Tensor<[1, 2, 3]>`
2. `Tensor<20>` with `TensorView<(1, 2, 3)>`

- Combining two tensors must be done via a `TensorView` (this is the only way while preventing ambiguity).

We might have to stick with a fully generic `Tensor<I>`

---

> [!NOTE] Oct 17: The compiler seems to not be powerful enough to deduce that `Tensor<{ H * W * D }, _>` is sized, so I'm adding in `where Tensor<{ H * W * D }, _>: Sized` clauses in most places. Also, sidenote: this where clause explains the other cryptic where clause I pointed out in a commit message yesterday.

> [!NOTE] Oct 17: In the current implementation of `tensor!`, you get a stack overflow message; however, this is not due to nested macro expansion but actually is a problem _at runtime_ - allocation! As everything is stack allocated, it shouldn't be a surprise `tensor!(2, 3, 89, 200, 20)` overflows the stack but `tensor!(2, 3, 89, 2, 2, 4, 9)` doesn't. Should I just turn every `[T; N]` into a `Box<[T; N]`?

> [!NOTE] Oct 17: So I gave up implementing `Index<[usize; D]> for Tensor<N, D, Shape>` for now.

### Oct 18

When attempting to return ..., the `ops::Index` trait only allows for indexing that returns a `&Self::Output`, so you can't return a reference-counted

```rs
impl<const N: usize, const D: usize, Shape> ops::Index<usize> for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized,
{
    type Output = <Shape as ops::Index<usize>>::Output;

    fn index(&self, index: usize) -> &Self::Output {
        (unsafe { transmute::<&[f64], Shape>(&self.data) })[index]
    }
}
```

On another note this day, the issues that make creating such an API so hard are the following features of Rust:

- non-variadic (const) generic parameters (and subsequently no way to iterate over them)

Due to this, I decided to encode shape information in three generic parameters: `const N: usize` and `const D: usize` and `Shape`.

The reason for three parameters is because Rust offers no way to manipulate between one another once the tensor struct has been initialized. Exercise: try doing this (set a timer for a good 6 minutes). Exploiting that, I've decided to encode all valuable information at the time of constructing the tensor.

While you are reading the following, make sure all of this makes sense logically to be part of a tensor's type (an example of something that shouldn't: actual data).

1. `const N: usize` — how many elements in total the tensor must contain
2. `const D: usize` — dimension of the tensor
3. `Shape` — unrestricted generic parameter (examples: `[f64; 2]`, `[[[f64; 4]; 3]; 42]`)

Very big alarm (one of the following types should not exist):

```rs
    Tensor<{ IH * IW * IC }, 3, shape_ty!(IH, IW, IC)>: Sized,
    Tensor<{ KH * KW * IC }, 3, shape_ty!(KH, KW, IC)>: Sized,
```
