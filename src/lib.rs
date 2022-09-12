#![feature(proc_macro_diagnostic)]
mod function;
mod tensor_type;
use function::Args;
use proc_macro::TokenStream;
use quote::quote;
use syn::fold::Fold;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Item, ItemFn, Result};

#[proc_macro_attribute]
pub fn tensor_check_fn(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemFn);

    // Parse the list of variables the user wanted to print.
    let mut args = Args::new();
    // use a syntax tree traversal to transform the function body.
    let output = args.fold_item_fn(input);

    // hand the resulting function body back to the compiler.
    TokenStream::from(quote!(#output))
}

struct Items {
    items: Vec<Item>,
}

impl Parse for Items {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut items = vec![];
        while !input.is_empty() {
            items.push(input.parse()?);
        }

        Ok(Self { items })
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
                    let mut args = Args::new();
                    // use a syntax tree traversal to transform the function body.
                    args.fold_item_fn(function.clone())
                }
                it => todo!("Item {it:?}"),
            }
        })
        .collect();
    // hand the resulting function body back to the compiler.
    TokenStream::from(quote!(#(#output)*))
}
