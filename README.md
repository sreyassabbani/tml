# `tml` _/ˈtɪ̆ml̩/_

An **experimental** type-safe machine learning library. Implementing concepts from scratch is cool, so that's what I'm doing here.

### Philosophy

A _lot_ of thought went into developer experience/API design, internal data flow, and performance. I'm starting to develop a set of principles in my everyday work nowadays, with the "parent principle" resting on the library user (everything rests on the end user): <ins>make the default the correct choice</ins>. I don't want to sell a religion, but the following is a tentative list of the best basis (orthogonal axes) that spans good software.

- make invalid states unrepresentable
  - special case: [parse, don't validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/)
- design for local reasoning
- fast[^1]

To be the best learning experience for me, I'm adding another principle:

- no external math libraries

This will limit the performance of the library, but this is a learning experience anyway. 

For more on the development philosophy, goals, and design decisions for this library, see [DESIGN](DESIGN.md).

### Development

1. Clone the repo.
2. Requirements: `rustc` nightly.

Notice that in order to run this, I have `rust-toolchain.toml` set to `toolchain.channel = "nightly"`. You may also opt to have control of every commands by selecting `cargo +nightly ...`.


[^1]: yes. and also we don't talk about Rust compile times
