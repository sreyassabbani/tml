# `tml` _/ˈtɪ̆ml̩/_

An **experimental** type-safe machine learning library. Implementing concepts from scratch is cool, so that's what I'm doing here.

Currently, I'm working on building a somewhat fast[^1] Rust library from the ground up (yes, no external math libraries) with API design and "parse, don't validate" as a large focus.

- Multi-layer perceptrons
- Convolutional neural networks
- Automatic differentiation

And possibly a lot more in the future

### Philosophy

A _lot_ of thought went into developer experience/API design, internal data flow, and performance. For the development philosophy, goals, and design decisions for this library, see [DESIGN](DESIGN.md).

### Development

1. Clone the repo.
2. Requirements: `rustc` nightly.

Notice that in order to run this, I have `rust-toolchain.toml` set to `toolchain.channel = "nightly"`. You may also opt to have control of every commands by selecting `cargo +nightly ...`.


[^1]: yes. and also we don't talk about Rust compile times
