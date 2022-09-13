//! Start checking your tensors shapes and operations at compile time instead of failing randomly
//! at runtime.
//!
//! This is work in progress, mostly as a proof of concept.
//!
//! Aligned with the idea of correctness of Rust, the idea is to bring that
//! capacity as much as possible into Machine Learning. This crate relies entirely
//! on [`tch-rs`] which is a Rust wrapper on top of [pytorch](https://pytorch.org/)
//!
//! The type system for tensors is a complex topic [link1](http://nlp.seas.harvard.edu/NamedTensor)
//! [link2](https://github.com/LaurentMazare/tch-rs/issues/112).
//! Most importantly we need to be able to [solve algebraic equations on
//! types](https://github.com/LaurentMazare/tch-rs/issues/112#issuecomment-531521703)
//! in order to actually solve the problem. Broadcasting is another one (some functions
//! have "multiple" signature let's say, which is not allowed in Rust).
//! Afaik, even GATs won't allow to implement everything directly within Rust type system.
//!
//! This crate implements a small compiler type checker on top of rust as a proc_macro.
//! The advantage of this, is that there is extreme liberty in what actually goes in,
//! it does not depend on GATs or praying that some RFCs land in Rust itself. We
//! can also liberally be more accepting of the actual syntax (since we can rewrite the code
//! afterwards to actually fit the Rust compiler).
//! The biggest drawback is that a LOT of the type checker has to be reimplemented with
//! type detection, call detection so on and so force. The main thing that makes this approach possible
//! is that the amount of method calls and functions are actually relatively limited.
//!
//! Let me know if you find any example/case which *cannot* ever be solved by this approach.
//!
//! Examples of failures that can be caught with this approach
//!
//! We can infer the types of simple tensor creation then detect that element wise
//! addition will fail.
//! ```compile_fail
//!     use static_typing_tch::tensor_check;
//!     use tch::Tensor;
//!
//!     tensor_check!{
//!     // The dimensions do not match
//!     let a = Tensor::ones(&[2, 3], (Kind::Float, Device::Cpu));
//!     let b = Tensor::ones(&[3, 2], (Kind::Float, Device::Cpu));
//!
//!     let c = a + b;
//!     }
//! ```
//!
//! Here we define a function that acts on specific shapes, and our view argument is
//! actually wrong meaning we cannot use `addmm` afterwards. This example requires
//! an actual algebraic solver to work.
//! ```compile_fail
//!   tensor_check! {
//!   fn transformer_mlp(
//!       hidden_states: &Tensor<(B, S, H)>,
//!       dense: &Tensor<(H, H4)>,
//!       dense_bias: &Tensor<(U1, H4)>,
//!   ) -> Tensor<(B, S, H)> {
//!       let size = hidden_states.size();
//!       let hidden_states = hidden_states.view((-1, size[1])); // Error here
//!       let hidden_states = dense_bias.addmm(&hidden_states, &dense);
//!       let hidden_states = hidden_states.view((size[0], size[1], size[2]));
//!       hidden_states
//!   }
//!   }
//! ```
//!
//! [More examples](https://github.com/Narsil/static_typing_tch/tree/main/tests/failures)

#![feature(proc_macro_diagnostic)]

mod function;
mod tensor_type;
use crate::function::Signature;
use function::Args;
use proc_macro::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::fold::Fold;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Item, Result};

struct Items {
    items: Vec<Item>,
    fns: HashMap<Ident, Signature>,
}

impl Parse for Items {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut items = vec![];
        while !input.is_empty() {
            items.push(input.parse()?);
        }
        let fns = HashMap::new();

        Ok(Self { items, fns })
    }
}

#[proc_macro]
pub fn tensor_check(input: TokenStream) -> TokenStream {
    let mut items: Items = syn::parse(input).unwrap();
    let output: Vec<_> = items
        .items
        .iter_mut()
        .map(|item| {
            match item {
                Item::Fn(function) => {
                    // Parse the list of variables the user wanted to print.
                    let mut args = Args::new(items.fns.clone());
                    // use a syntax tree traversal to transform the function body.
                    let output = args.fold_item_fn(function.clone());
                    let name = function.sig.ident.clone();
                    items.fns.insert(name, args.signature());
                    output
                }
                it => todo!("Item {it:?}"),
            }
        })
        .collect();
    // hand the resulting function body back to the compiler.
    TokenStream::from(quote!(#(#output)*))
}
