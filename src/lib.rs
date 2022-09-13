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
