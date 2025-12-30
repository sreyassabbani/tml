use nn_utils::layerable::{LayerKind, Layerable};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Ident, LitInt, Token, parse_macro_input};

// Custom parsing for our network DSL
mod parsing {
    use super::*;
    use syn::parse::{Parse, ParseStream};

    fn parse_optional_usizes<const N: usize>(
        content: &syn::parse::ParseBuffer<'_>,
        defaults: [usize; N],
    ) -> syn::Result<[usize; N]> {
        let mut values = defaults;

        for slot in &mut values {
            if !content.peek(Token![,]) {
                break;
            }
            content.parse::<Token![,]>()?;
            *slot = content.parse::<LitInt>()?.base10_parse()?;
        }

        Ok(values)
    }

    #[derive(Debug, Clone)]
    pub struct LayerSpec {
        pub input: usize,
        pub kind: LayerKind,
    }

    impl LayerSpec {
        pub fn new(input: usize, kind: LayerKind) -> Self {
            Self { input, kind }
        }
    }

    impl Layerable for LayerSpec {
        fn input(&self) -> usize {
            self.input
        }

        fn kind(&self) -> LayerKind {
            self.kind.clone()
        }
    }

    pub struct NetworkDef {
        pub layers: Vec<LayerSpec>,
    }

    impl Parse for NetworkDef {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            input.parse::<Ident>()?;

            let content;
            ::syn::parenthesized!(content in input);
            let mut cur_size = content.parse::<LitInt>()?.base10_parse()?;

            input.parse::<Token![->]>()?;

            let mut layers = Vec::new();

            while !input.is_empty() {
                let layer_name: Ident = input.parse()?;

                match layer_name.to_string().as_str() {
                    "dense" => {
                        let content;
                        ::syn::parenthesized!(content in input);
                        let next_size = content.parse::<LitInt>()?.base10_parse()?;
                        layers.push(LayerSpec::new(
                            cur_size,
                            LayerKind::Dense { output: next_size },
                        ));

                        // resize network width
                        cur_size = next_size;
                    }
                    "relu" | "ReLU" => {
                        layers.push(LayerSpec::new(
                            cur_size,
                            LayerKind::ReLU { width: cur_size },
                        ));
                    }
                    "sigmoid" | "Sigmoid" => {
                        layers.push(LayerSpec::new(
                            cur_size,
                            LayerKind::Sigmoid { width: cur_size },
                        ));
                    }
                    "conv" | "Conv" => {
                        // parse parens with comma-separated ints; allow optional named args later
                        let content;
                        ::syn::parenthesized!(content in input);

                        // minimal syntax: conv(out, kernel) or conv(out, kernel, stride, pad)
                        let out_c: LitInt = content.parse()?;
                        let _comma = content.parse::<Token![,]>()?;
                        let k: LitInt = content.parse()?;

                        // parse optional stride, pad (or more) using a generic helper
                        let [stride, pad] = parse_optional_usizes(&content, [1, 0])?;

                        let out_channels = out_c.base10_parse()?;
                        layers.push(LayerSpec::new(
                            cur_size,
                            LayerKind::Conv {
                                out_channels,
                                kernel: k.base10_parse()?,
                                stride,
                                padding: pad,
                            },
                        ));

                        cur_size = out_channels;
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

            Ok(NetworkDef { layers })
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

fn generate_network(def: parsing::NetworkDef) -> TokenStream2 {
    let input_size = def.layers.first().map(|l| l.input()).unwrap_or(0);
    let layer_count = def.layers.len();

    // Calculate maximum buffer size needed
    let mut current_size = input_size;
    let mut max_size = input_size;
    let mut layer_io = Vec::with_capacity(layer_count);
    let mut layer_types = Vec::with_capacity(layer_count);

    for layer in &def.layers {
        let kind = layer.kind();

        let next_size = match kind {
            LayerKind::Dense { output } => output,
            LayerKind::ReLU { .. } | LayerKind::Sigmoid { .. } => current_size,
            LayerKind::Conv { out_channels, .. } => out_channels,
        };

        layer_io.push((current_size, next_size));

        let tokens = match kind {
            LayerKind::Dense { output } => {
                quote! { ::nn::network::DenseLayer<#current_size, #output> }
            }
            LayerKind::ReLU { .. } => quote! { ::nn::network::ReLU<#current_size> },
            LayerKind::Sigmoid { .. } => quote! { ::nn::network::Sigmoid<#current_size> },
            LayerKind::Conv { .. } => quote! { ::nn::network::Conv<#current_size> },
        };

        layer_types.push(tokens);

        max_size = max_size.max(next_size);
        current_size = next_size;
    }

    let output_size = current_size;

    // Generate forward pass with buffer reuse
    let mut forward_calls = Vec::new();
    let mut use_buf_a = true;

    for (i, (in_size, out_size)) in layer_io.iter().enumerate() {
        let layer_idx = ::syn::Index::from(i);
        let (input_buf, output_buf) = if use_buf_a {
            (quote! { &self._buf_a }, quote! { &mut self._buf_b })
        } else {
            (quote! { &self._buf_b }, quote! { &mut self._buf_a })
        };

        // forward_calls.push(quote! {
        //     self.layers.#layer_idx.forward(
        //         <&[f32; #current_size]>::try_into(#input_buf[..#current_size]).unwrap(),
        //         <&mut [f32; #current_size]>::try_into(&mut #output_buf[..#current_size]).unwrap(),
        //     );
        // });

        forward_calls.push(quote! {
            self.layers.#layer_idx.forward(
                #input_buf[..#in_size],
                #output_buf[..#out_size],
            );
        });

        use_buf_a = !use_buf_a;
    }

    // Generate layer initializations
    let layer_inits = layer_types.iter().map(|layer_type| {
        quote! { <#layer_type>::init() }
    });

    let final_buffer = if (layer_count % 2) == 1 {
        quote! { self._buf_b }
    } else {
        quote! { self._buf_a }
    };

    quote! {
        {
            #[derive(Debug)]
            struct Network<Layers> {
                layers: Layers,
                // Double buffering approach with fixed-size boxes
                _buf_a: Box<[f32; #max_size]>,
                _buf_b: Box<[f32; #max_size]>,
            }

            struct NetworkWorkspace {

            }

            impl Network<(#(#layer_types,)*)> {
                pub fn new() -> Self {
                    Network {
                        layers: (#(#layer_inits,)*),
                        _buf_a: Box::new([Default::default(); #max_size]),
                        _buf_b: Box::new([Default::default(); #max_size]),
                    }
                }

                pub fn forward_with_workspace(&self, input: &[f32; #input_size], workspace: &mut NetworkWorkspace) -> [f32; #output_size] {
                    // used to be forward<I: AsRef<[f32; #input_size]>>(... input: I)

                    // Copy input to first buffer
                    // self._buf_a[..#input_size].copy_from_slice(input);

                    // Run forward pass with ping-pong buffers
                    // #(#forward_calls)*;

                    // Extract result from final buffer
                    let mut result = [0.0; #output_size];
                    result.copy_from_slice(&(#final_buffer)[..#output_size]);
                    result
                }

                pub fn forward(&self, input: &[f32; #input_size]) -> [f32; #output_size] {
                    // Copy input to first buffer
                    self.buffers.0 = *input;

                    // Run forward pass
                    #(#forward_calls)*

                    // Return final buffer
                    #final_buffer
                    // [0.0; #output_size]
                }

                pub fn train<D: AsRef<[[f32; #input_size]]>, T: AsRef<[[f32; #output_size]]>>(&mut self, data: D, targets: T) {
                    // Loop over each case
                    let targets = targets.as_ref().iter();
                    let data = data.as_ref().iter();

                    for (input, target) in data.zip(targets) {
                        let out = self.forward(input);
                        let loss: f32 = out.iter().zip(target.iter()).map(|(o, t)| (o - t).powi(2)).sum();
                        // sum (y hat - y)^2

                        // for layer in self.layers.iter() {

                        // }
                    }

                    // Training implementation
                }
            }

            Network::<(#(#layer_types,)*)>::new()
        }
    }
}
