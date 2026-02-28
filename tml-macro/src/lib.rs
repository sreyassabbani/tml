use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Expr, Ident, Token, parse_macro_input};

// Custom parsing for our network DSL
mod parsing {
    use super::*;
    use syn::parse::{Parse, ParseStream};

    #[derive(Debug, Clone)]
    pub enum InputShape {
        Vec {
            n: TokenStream2,
        },
        Image {
            c: TokenStream2,
            h: TokenStream2,
            w: TokenStream2,
        },
    }

    fn parse_expr_list(content: &syn::parse::ParseBuffer<'_>) -> syn::Result<Vec<Expr>> {
        let mut values = Vec::new();
        if content.is_empty() {
            return Ok(values);
        }

        values.push(content.parse::<Expr>()?);
        while content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
            if content.is_empty() {
                break;
            }
            values.push(content.parse::<Expr>()?);
        }

        Ok(values)
    }

    #[derive(Debug, Clone)]
    pub enum LayerSpecKind {
        Dense {
            output: TokenStream2,
        },
        ReLU,
        Sigmoid,
        Flatten,
        Conv {
            out_channels: TokenStream2,
            kernel_h: TokenStream2,
            kernel_w: TokenStream2,
            stride: TokenStream2,
            padding: TokenStream2,
        },
    }

    #[derive(Debug, Clone)]
    pub struct LayerSpec {
        pub kind: LayerSpecKind,
    }

    pub struct NetworkDef {
        pub input: InputShape,
        pub layers: Vec<LayerSpec>,
    }

    impl Parse for NetworkDef {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let _name: Ident = input.parse()?;

            let content;
            ::syn::parenthesized!(content in input);
            let input_dims = parse_expr_list(&content)?;
            let input_shape = match input_dims.as_slice() {
                [n] => InputShape::Vec { n: quote! { #n } },
                [c, h, w] => InputShape::Image {
                    c: quote! { #c },
                    h: quote! { #h },
                    w: quote! { #w },
                },
                _ => {
                    return Err(::syn::Error::new(
                        content.span(),
                        "input must be (N) or (C, H, W)",
                    ));
                }
            };

            input.parse::<Token![->]>()?;

            let mut layers = Vec::new();

            while !input.is_empty() {
                let layer_name: Ident = input.parse()?;

                match layer_name.to_string().as_str() {
                    "dense" => {
                        let content;
                        ::syn::parenthesized!(content in input);
                        let next_size: Expr = content.parse()?;
                        layers.push(LayerSpec {
                            kind: LayerSpecKind::Dense {
                                output: quote! { #next_size },
                            },
                        });
                    }
                    "relu" | "ReLU" => {
                        layers.push(LayerSpec {
                            kind: LayerSpecKind::ReLU,
                        });
                    }
                    "sigmoid" | "Sigmoid" => {
                        layers.push(LayerSpec {
                            kind: LayerSpecKind::Sigmoid,
                        });
                    }
                    "flatten" | "Flatten" => {
                        layers.push(LayerSpec {
                            kind: LayerSpecKind::Flatten,
                        });
                    }
                    "conv" | "Conv" => {
                        let content;
                        ::syn::parenthesized!(content in input);
                        let args = parse_expr_list(&content)?;

                        let kind = match args.as_slice() {
                            [out_channels, kernel] => LayerSpecKind::Conv {
                                out_channels: quote! { #out_channels },
                                kernel_h: quote! { #kernel },
                                kernel_w: quote! { #kernel },
                                stride: quote! { 1 },
                                padding: quote! { 0 },
                            },
                            [out_channels, kernel, stride] => LayerSpecKind::Conv {
                                out_channels: quote! { #out_channels },
                                kernel_h: quote! { #kernel },
                                kernel_w: quote! { #kernel },
                                stride: quote! { #stride },
                                padding: quote! { 0 },
                            },
                            [out_channels, kernel, stride, padding] => LayerSpecKind::Conv {
                                out_channels: quote! { #out_channels },
                                kernel_h: quote! { #kernel },
                                kernel_w: quote! { #kernel },
                                stride: quote! { #stride },
                                padding: quote! { #padding },
                            },
                            [out_channels, kernel_h, kernel_w, stride, padding] => {
                                LayerSpecKind::Conv {
                                    out_channels: quote! { #out_channels },
                                    kernel_h: quote! { #kernel_h },
                                    kernel_w: quote! { #kernel_w },
                                    stride: quote! { #stride },
                                    padding: quote! { #padding },
                                }
                            }
                            _ => {
                                return Err(::syn::Error::new(
                                    content.span(),
                                    "conv expects (out, k) | (out, k, stride) | (out, k, stride, pad) | (out, k_h, k_w, stride, pad)",
                                ));
                            }
                        };

                        layers.push(LayerSpec { kind });
                    }
                    "output" => break,
                    _ => return Err(::syn::Error::new(layer_name.span(), "Unknown layer type")),
                }

                if !input.is_empty() && !input.peek(Token![->]) {
                    break;
                }

                if input.peek(Token![->]) {
                    input.parse::<Token![->]>()?;
                }
            }

            Ok(NetworkDef {
                input: input_shape,
                layers,
            })
        }
    }
}

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let network_def = parse_macro_input!(input as parsing::NetworkDef);

    // Generate the network code
    let generated = generate_network(network_def);

    generated.into()
}

#[derive(Clone, Debug)]
enum ShapeSpec {
    Vec {
        n: TokenStream2,
    },
    Image {
        c: TokenStream2,
        h: TokenStream2,
        w: TokenStream2,
    },
}

impl ShapeSpec {
    fn size_expr(&self) -> TokenStream2 {
        match self {
            ShapeSpec::Vec { n } => quote! { #n },
            ShapeSpec::Image { c, h, w } => quote! { #c * #h * #w },
        }
    }
}

fn max_expr(exprs: Vec<TokenStream2>) -> TokenStream2 {
    let mut iter = exprs.into_iter();
    let first = iter.next().unwrap_or_else(|| quote! { 0 });
    iter.fold(first, |acc, expr| quote! { __nn_max(#acc, #expr) })
}

fn generate_network(def: parsing::NetworkDef) -> TokenStream2 {
    let input_shape = match def.input {
        parsing::InputShape::Vec { n } => ShapeSpec::Vec { n },
        parsing::InputShape::Image { c, h, w } => ShapeSpec::Image { c, h, w },
    };

    let layer_count = def.layers.len();
    let input_size_expr = input_shape.size_expr();

    let mut current_shape = input_shape;
    let mut layer_io = Vec::with_capacity(layer_count);
    let mut layer_types = Vec::with_capacity(layer_count);
    let mut layer_out_sizes = Vec::with_capacity(layer_count);
    let mut conv_checks = Vec::with_capacity(layer_count);

    for layer in &def.layers {
        let in_size = current_shape.size_expr();
        let (next_shape, layer_type) = match &layer.kind {
            parsing::LayerSpecKind::Dense { output } => match current_shape {
                ShapeSpec::Vec { n } => (
                    ShapeSpec::Vec { n: output.clone() },
                    quote! { ::tml::network::DenseLayer<{ #n }, { #output }> },
                ),
                ShapeSpec::Image { .. } => {
                    return quote! {
                        ::core::compile_error!("dense expects a vector input; add flatten before dense");
                    };
                }
            },
            parsing::LayerSpecKind::ReLU => {
                let size = current_shape.size_expr();
                (current_shape, quote! { ::tml::network::ReLU<{ #size }> })
            }
            parsing::LayerSpecKind::Sigmoid => {
                let size = current_shape.size_expr();
                (current_shape, quote! { ::tml::network::Sigmoid<{ #size }> })
            }
            parsing::LayerSpecKind::Flatten => {
                let size = current_shape.size_expr();
                (
                    ShapeSpec::Vec { n: size.clone() },
                    quote! { ::tml::network::Flatten<{ #size }> },
                )
            }
            parsing::LayerSpecKind::Conv {
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => match &current_shape {
                ShapeSpec::Image { c, h, w } => {
                    conv_checks.push(quote! {
                        const _: () = {
                            if !(::tml::conv::conv_out_dim(#h, #padding, #kernel_h, #stride) > 0) {
                                panic!("conv: invalid height (check input H, kernel, stride, padding)");
                            }
                            if !(::tml::conv::conv_out_dim(#w, #padding, #kernel_w, #stride) > 0) {
                                panic!("conv: invalid width (check input W, kernel, stride, padding)");
                            }
                        };
                    });
                    let out_h =
                        quote! { ::tml::conv::conv_out_dim(#h, #padding, #kernel_h, #stride) };
                    let out_w =
                        quote! { ::tml::conv::conv_out_dim(#w, #padding, #kernel_w, #stride) };
                    (
                        ShapeSpec::Image {
                            c: out_channels.clone(),
                            h: out_h,
                            w: out_w,
                        },
                        quote! {
                            ::tml::conv::Conv<{ #w }, { #h }, { #c }, { #kernel_h }, { #kernel_w }, { #out_channels }, { #stride }, { #padding }>
                        },
                    )
                }
                ShapeSpec::Vec { .. } => {
                    return quote! {
                        ::core::compile_error!("conv expects a (C, H, W) input shape");
                    };
                }
            },
        };

        let out_size = next_shape.size_expr();
        layer_io.push((in_size, out_size.clone()));
        layer_types.push(layer_type);
        layer_out_sizes.push(out_size);
        current_shape = next_shape;
    }

    let output_size_expr = current_shape.size_expr();
    let max_size_expr = max_expr(
        std::iter::once(input_size_expr.clone())
            .chain(layer_out_sizes.iter().cloned())
            .collect(),
    );

    let layer_inits = layer_types.iter().map(|layer_type| {
        quote! { <#layer_type>::init() }
    });

    let act_idents = (0..layer_count)
        .map(|i| format_ident!("act_{}", i))
        .collect::<Vec<_>>();
    let grad_idents = (0..layer_count)
        .map(|i| format_ident!("grad_{}", i))
        .collect::<Vec<_>>();

    let activation_fields = act_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box<[::tml::Float; #out_size]> }
        });
    let gradient_fields = grad_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box<[::tml::Float; #out_size]> }
        });

    let activation_inits = act_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box::new([Default::default(); #out_size]) }
        });
    let gradient_inits = grad_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box::new([Default::default(); #out_size]) }
        });

    let mut forward_calls_ws = Vec::new();
    for (i, (in_size, out_size)) in layer_io.iter().enumerate() {
        let layer_idx = ::syn::Index::from(i);
        let input_ref = if i == 0 {
            quote! { workspace.act_input.as_ref() }
        } else {
            let prev = &act_idents[i - 1];
            quote! { workspace.#prev.as_ref() }
        };
        let out_ident = &act_idents[i];
        forward_calls_ws.push(quote! {
            ::tml::network::Layer::<{ #in_size }, { #out_size }>::forward(
                &layers.#layer_idx,
                #input_ref,
                workspace.#out_ident.as_mut(),
            );
        });
    }

    let mut forward_calls_buf = Vec::new();
    let mut use_buf_a = true;
    for (i, (in_size, out_size)) in layer_io.iter().enumerate() {
        let layer_idx = ::syn::Index::from(i);
        let (input_buf, output_buf) = if use_buf_a {
            (quote! { buf_a }, quote! { buf_b })
        } else {
            (quote! { buf_b }, quote! { buf_a })
        };

        forward_calls_buf.push(quote! {
            let input_arr: &[::tml::Float; #in_size] =
                <&[::tml::Float; #in_size]>::try_from(&#input_buf[..#in_size])
                    .expect("invalid input buffer size");
            let output_arr: &mut [::tml::Float; #out_size] =
                <&mut [::tml::Float; #out_size]>::try_from(&mut #output_buf[..#out_size])
                    .expect("invalid output buffer size");
            ::tml::network::Layer::<{ #in_size }, { #out_size }>::forward(
                &self.layers.#layer_idx,
                input_arr,
                output_arr,
            );
        });

        use_buf_a = !use_buf_a;
    }

    let final_buffer = if (layer_count % 2) == 1 {
        quote! { buf_b }
    } else {
        quote! { buf_a }
    };

    let mut backward_calls = Vec::new();
    for (i, (in_size, out_size)) in layer_io.iter().enumerate().rev() {
        let layer_idx = ::syn::Index::from(i);
        let input_act = if i == 0 {
            quote! { workspace.act_input.as_ref() }
        } else {
            let prev = &act_idents[i - 1];
            quote! { workspace.#prev.as_ref() }
        };
        let output_act = &act_idents[i];
        let output_grad = &grad_idents[i];
        let input_grad = if i == 0 {
            quote! { workspace.grad_input.as_mut() }
        } else {
            let prev = &grad_idents[i - 1];
            quote! { workspace.#prev.as_mut() }
        };

        backward_calls.push(quote! {
            ::tml::network::Layer::<{ #in_size }, { #out_size }>::backward(
                &mut layers.#layer_idx,
                #input_act,
                workspace.#output_act.as_ref(),
                workspace.#output_grad.as_ref(),
                #input_grad,
                lr,
            );
        });
    }

    let last_act_ident = act_idents
        .last()
        .cloned()
        .unwrap_or_else(|| format_ident!("act_input"));
    let last_grad_ident = grad_idents
        .last()
        .cloned()
        .unwrap_or_else(|| format_ident!("grad_input"));

    quote! {
        {
            const fn __nn_max(a: usize, b: usize) -> usize {
                if a > b { a } else { b }
            }

            const INPUT_SIZE: usize = #input_size_expr;
            const OUTPUT_SIZE: usize = #output_size_expr;
            const MAX_BUF: usize = #max_size_expr;

            #(#conv_checks)*

            #[derive(Debug)]
            struct Network<Layers> {
                layers: Layers,
                _buf_a: Box<[::tml::Float; MAX_BUF]>,
                _buf_b: Box<[::tml::Float; MAX_BUF]>,
            }

            #[derive(Debug)]
            struct NetworkWorkspace {
                act_input: Box<[::tml::Float; INPUT_SIZE]>,
                #(#activation_fields,)*
                grad_input: Box<[::tml::Float; INPUT_SIZE]>,
                #(#gradient_fields,)*
            }

            impl NetworkWorkspace {
                fn new() -> Self {
                    Self {
                        act_input: Box::new([Default::default(); INPUT_SIZE]),
                        #(#activation_inits,)*
                        grad_input: Box::new([Default::default(); INPUT_SIZE]),
                        #(#gradient_inits,)*
                    }
                }
            }

            impl Default for NetworkWorkspace {
                fn default() -> Self {
                    Self::new()
                }
            }

            impl Network<(#(#layer_types,)*)> {
                pub fn new() -> Self {
                    Network {
                        layers: (#(#layer_inits,)*),
                        _buf_a: Box::new([Default::default(); MAX_BUF]),
                        _buf_b: Box::new([Default::default(); MAX_BUF]),
                    }
                }

                pub fn workspace(&self) -> NetworkWorkspace {
                    NetworkWorkspace::new()
                }

                pub fn inference(
                    &self,
                    input: &[::tml::Float; INPUT_SIZE],
                ) -> [::tml::Float; OUTPUT_SIZE] {
                    let mut workspace = NetworkWorkspace::new();
                    let output = self.inference_with_workspace(input, &mut workspace);
                    let mut result = [0.0 as ::tml::Float; OUTPUT_SIZE];
                    result.copy_from_slice(output);
                    result
                }

                fn inference_with_workspace_layers<'a>(
                    layers: &(#(#layer_types,)*),
                    input: &[::tml::Float; INPUT_SIZE],
                    workspace: &'a mut NetworkWorkspace,
                ) -> &'a [::tml::Float; OUTPUT_SIZE] {
                    workspace.act_input.copy_from_slice(input);
                    #(#forward_calls_ws)*
                    &workspace.#last_act_ident
                }

                pub fn inference_with_workspace<'a>(
                    &self,
                    input: &[::tml::Float; INPUT_SIZE],
                    workspace: &'a mut NetworkWorkspace,
                ) -> &'a [::tml::Float; OUTPUT_SIZE] {
                    Self::inference_with_workspace_layers(&self.layers, input, workspace)
                }

                pub fn inference_in_place(
                    &mut self,
                    input: &[::tml::Float; INPUT_SIZE],
                ) -> [::tml::Float; OUTPUT_SIZE] {
                    let (buf_a, buf_b) = (&mut self._buf_a[..], &mut self._buf_b[..]);

                    buf_a[..INPUT_SIZE].copy_from_slice(input);

                    #(#forward_calls_buf)*

                    let mut result = [0.0 as ::tml::Float; OUTPUT_SIZE];
                    result.copy_from_slice(&#final_buffer[..OUTPUT_SIZE]);
                    result
                }

                #[deprecated(note = "use inference")]
                pub fn forward(
                    &self,
                    input: &[::tml::Float; INPUT_SIZE],
                ) -> [::tml::Float; OUTPUT_SIZE] {
                    self.inference(input)
                }

                #[deprecated(note = "use inference_with_workspace")]
                pub fn forward_with_workspace<'a>(
                    &self,
                    input: &[::tml::Float; INPUT_SIZE],
                    workspace: &'a mut NetworkWorkspace,
                ) -> &'a [::tml::Float; OUTPUT_SIZE] {
                    self.inference_with_workspace(input, workspace)
                }

                #[deprecated(note = "use inference_in_place")]
                pub fn forward_in_place(
                    &mut self,
                    input: &[::tml::Float; INPUT_SIZE],
                ) -> [::tml::Float; OUTPUT_SIZE] {
                    self.inference_in_place(input)
                }

                fn backward_with_workspace_layers(
                    layers: &mut (#(#layer_types,)*),
                    workspace: &mut NetworkWorkspace,
                    lr: ::tml::Float,
                ) {
                    #(#backward_calls)*
                }

                fn backward_with_workspace(&mut self, workspace: &mut NetworkWorkspace, lr: ::tml::Float) {
                    Self::backward_with_workspace_layers(&mut self.layers, workspace, lr);
                }

                fn train_step_layers(
                    layers: &mut (#(#layer_types,)*),
                    input: &[::tml::Float; INPUT_SIZE],
                    target: &[::tml::Float; OUTPUT_SIZE],
                    mut workspace: NetworkWorkspace,
                    lr: ::tml::Float,
                ) -> (NetworkWorkspace, ::tml::Float) {
                    Self::inference_with_workspace_layers(layers, input, &mut workspace);
                    let loss = ::tml::network::mse_loss(
                        workspace.#last_act_ident.as_ref(),
                        target,
                        workspace.#last_grad_ident.as_mut(),
                    );
                    Self::backward_with_workspace_layers(layers, &mut workspace, lr);
                    (workspace, loss)
                }

                fn train_step(
                    &mut self,
                    input: &[::tml::Float; INPUT_SIZE],
                    target: &[::tml::Float; OUTPUT_SIZE],
                    workspace: NetworkWorkspace,
                    lr: ::tml::Float,
                ) -> (NetworkWorkspace, ::tml::Float) {
                    Self::train_step_layers(&mut self.layers, input, target, workspace, lr)
                }

                pub fn train_with<
                    D: AsRef<[[::tml::Float; INPUT_SIZE]]>,
                    T: AsRef<[[::tml::Float; OUTPUT_SIZE]]>,
                >(
                    &mut self,
                    data: D,
                    targets: T,
                    config: ::tml::network::TrainConfig,
                ) -> ::tml::Float {
                    let data = data.as_ref();
                    let targets = targets.as_ref();
                    assert_eq!(
                        data.len(),
                        targets.len(),
                        "data/target length mismatch: {} vs {}",
                        data.len(),
                        targets.len()
                    );
                    if data.is_empty() || config.epochs == 0 {
                        return 0.0;
                    }

                    let mut workspace = NetworkWorkspace::new();
                    let mut total_loss = 0.0;
                    let mut steps = 0usize;
                    let layers = &mut self.layers;

                    for _ in 0..config.epochs {
                        for (input, target) in data.iter().zip(targets.iter()) {
                            let (next_workspace, loss) = Self::train_step_layers(
                                layers,
                                input,
                                target,
                                workspace,
                                config.lr,
                            );
                            workspace = next_workspace;
                            total_loss += loss;
                            steps += 1;
                        }
                    }

                    total_loss / steps as ::tml::Float
                }

                pub fn fit_with(
                    &mut self,
                    samples: &[::tml::Sample<INPUT_SIZE, OUTPUT_SIZE>],
                    config: ::tml::network::TrainConfig,
                ) -> ::tml::Float {
                    if samples.is_empty() || config.epochs == 0 {
                        return 0.0;
                    }

                    let mut workspace = NetworkWorkspace::new();
                    let mut total_loss = 0.0;
                    let mut steps = 0usize;
                    let layers = &mut self.layers;

                    for _ in 0..config.epochs {
                        for sample in samples {
                            let (next_workspace, loss) = Self::train_step_layers(
                                layers,
                                &sample.input,
                                &sample.target,
                                workspace,
                                config.lr,
                            );
                            workspace = next_workspace;
                            total_loss += loss;
                            steps += 1;
                        }
                    }

                    total_loss / steps as ::tml::Float
                }

                pub fn fit(
                    &mut self,
                    samples: &[::tml::Sample<INPUT_SIZE, OUTPUT_SIZE>],
                ) -> ::tml::Float {
                    self.fit_with(samples, ::tml::network::TrainConfig::default())
                }

                pub fn train<
                    D: AsRef<[[::tml::Float; INPUT_SIZE]]>,
                    T: AsRef<[[::tml::Float; OUTPUT_SIZE]]>,
                >(
                    &mut self,
                    data: D,
                    targets: T,
                ) -> ::tml::Float {
                    self.train_with(data, targets, ::tml::network::TrainConfig::default())
                }
            }

            Network::<(#(#layer_types,)*)>::new()
        }
    }
}
