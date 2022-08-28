#![feature(proc_macro_diagnostic)]
use proc_macro::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::fold::{self, Fold};
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, parse_quote, AngleBracketedGenericArguments, Expr, ExprArray, ExprCall,
    ExprPath, ExprTuple, FnArg, GenericArgument, Ident, ItemFn, Lit, Local, Pat, PatType,
    PathArguments, ReturnType, Stmt, Type, TypeParamBound, TypePath,
};

struct Args {
    vars: HashMap<Ident, DetectedType>,
    return_type: DetectedType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DetectedType {
    NotDetected,
    NotTensor,
    Inferred(TensorType),
    Declared(TensorType),
}

impl Args {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            return_type: DetectedType::NotDetected,
        }
    }
}

impl Args {
    fn let_and_print(&mut self, local: Local) -> Stmt {
        let Local { pat, init, .. } = local;
        let expr = *init.unwrap().1;

        let ident = match pat {
            Pat::Ident(ref p) => &p.ident,
            _ => unreachable!(),
        };
        match &expr {
            Expr::Call(m) => self.maybe_assign_type(&ident, m),
            Expr::Binary(b) => {
                let detected_left = self.get_type(&b.left);
                let detected_right = self.get_type(&b.right);
                if !compatible(&detected_left, &detected_right) {
                    let error = format!(
                        "Tensor mismatch {:?} is incompatible with {:?}",
                        detected_left, detected_right
                    );
                    b.span().unwrap().error(error).emit();
                }
            }
            s => println!("Other {s:?}"),
        }
        let init = self.fold_expr(expr);
        parse_quote! {
            let #pat = {
                #[allow(unused_mut)]
                let #pat = #init;
                println!(concat!(stringify!(#ident), " = {:?}"), #ident);
                #ident
            };
        }
    }

    fn get_type(&self, expr: &Expr) -> DetectedType {
        match expr {
            Expr::Path(path) => {
                if path.path.segments.len() == 1 {
                    self.vars
                        .get(&path.path.segments[0].ident)
                        .unwrap_or(&DetectedType::NotDetected)
                        .clone()
                } else {
                    todo!()
                }
            }
            _ => todo!(),
        }
    }

    fn maybe_assign_type(&mut self, ident: &Ident, call: &ExprCall) {
        let detected_type = self.detect_type_call(call);
        if let DetectedType::Inferred(type_) = detected_type {
            self.assign_type(&ident, type_);
        }
    }

    fn detect_type_call(&self, call: &ExprCall) -> DetectedType {
        let args = &call.args;
        match &*call.func {
            Expr::Path(expr) => {
                if &expr.path.segments[0].ident.to_string() == "Tensor" {
                    match &expr.path.segments[1].ident.to_string()[..] {
                        "ones" | "zeros" => {
                            let shape: Vec<Dim> = self.detect_shape(&args[0]);
                            let (kind, device) = self.detect_kind_device(&args[1]);
                            DetectedType::Inferred(TensorType::new(shape, kind, device))
                        }
                        "cat" => {
                            call.func
                                .span()
                                .unwrap()
                                .warning("Cat not handled well yet")
                                .emit();
                            assert!(args.len() == 2);
                            DetectedType::Inferred(self.detect_cat_type(&args[0], &args[1]))
                        }
                        s => {
                            call.func
                                .span()
                                .unwrap()
                                .error(format!("{s} not handled"))
                                .emit();
                            DetectedType::NotDetected
                        }
                    }
                } else {
                    DetectedType::NotTensor
                }
            }
            _ => todo!(),
        }
    }

    fn detect_cat_type(&self, tensors: &Expr, dim: &Expr) -> TensorType {
        let dim: i64 = match dim {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Int(int) => int.base10_parse().unwrap(),
                _ => panic!("Can't handle non int cat"),
            },
            _ => panic!("Can't handle dynamic cat yet."),
        };
        let types: Vec<TensorType> = match tensors {
            Expr::Reference(reference) => match &*reference.expr {
                Expr::Array(array) => array
                    .elems
                    .iter()
                    .map(|item| self.detect_type(item))
                    .collect(),
                _ => panic!("cat is not supposed to be call that way"),
            },
            _ => panic!("cat is not supposed to be call that way"),
        };
        println!("Tensors {types:#?}");

        todo!("Need to finish cat");
    }

    fn detect_type(&self, expr: &Expr) -> TensorType {
        match expr {
            Expr::Path(path) => {
                assert!(path.path.segments.len() == 1);
                let name = &path.path.segments[0].ident;
                match self.vars.get(name) {
                    Some(DetectedType::Declared(tensor_type)) => tensor_type.clone(),
                    Some(DetectedType::Inferred(tensor_type)) => tensor_type.clone(),
                    _ => {
                        name.span()
                            .unwrap()
                            .error("Couldn't get the tensor type for this variable")
                            .emit();
                        unreachable!();
                    }
                }
            }
            _ => todo!(),
        }
    }

    fn detect_shape(&self, expr: &Expr) -> Vec<Dim> {
        match expr {
            Expr::Reference(expr_ref) => match &*expr_ref.expr {
                Expr::Array(ExprArray { elems, .. }) => elems
                    .iter()
                    .filter_map(|elem| match elem {
                        Expr::Lit(expr_lit) => match &expr_lit.lit {
                            Lit::Int(lit_int) => Some(Dim::Value(lit_int.base10_parse().unwrap())),
                            _ => todo!(),
                        },
                        _ => None,
                    })
                    .collect(),
                _ => todo!(),
            },
            _ => todo!(),
        }
    }

    fn detect_kind_device(&self, expr: &Expr) -> (Kind, Device) {
        match expr {
            Expr::Tuple(ExprTuple { elems, .. }) => {
                assert!(elems.len() == 2);
                let kind = match &elems[0] {
                    Expr::Path(ExprPath { path, .. }) => {
                        if &path.segments[1].ident.to_string() == "Float" {
                            Kind::Float
                        } else {
                            todo!()
                        }
                    }
                    _ => todo!(),
                };
                let device = match &elems[1] {
                    Expr::Path(ExprPath { path, .. }) => {
                        if &path.segments[1].ident.to_string() == "Cpu" {
                            Device::Cpu
                        } else {
                            todo!()
                        }
                    }
                    _ => todo!(),
                };
                (kind, device)
            }
            _ => todo!(),
        }
    }

    fn detect_type_arg(&self, arg: FnArg) -> (Ident, Type, DetectedType) {
        match arg {
            FnArg::Typed(PatType { pat, ty, .. }) => {
                let name = match *pat {
                    Pat::Ident(ident) => ident.ident,
                    _ => todo!(),
                };
                let mut ty = ty.clone();
                let detected_type = self.detect_type_type(&mut *ty);
                (name, *ty, detected_type)
            }
            _ => todo!(),
        }
    }

    fn detect_type_type(&self, ty: &mut Type) -> DetectedType {
        match ty {
            Type::Path(TypePath { ref mut path, .. }) => {
                let mut segment = &mut path.segments[0];
                if segment.ident.to_string() == "Tensor" {
                    let detected_type = match &mut segment.arguments {
                        PathArguments::AngleBracketed(angle) => self.detect_type_tuple(angle),
                        PathArguments::None => {
                            segment
                                .ident
                                .span()
                                .unwrap()
                                .error("Tensor needs to be annotated.")
                                .emit();
                            unreachable!()
                        }

                        _ => todo!(),
                    };
                    segment.arguments = PathArguments::None;
                    detected_type
                } else {
                    DetectedType::NotTensor
                }
            }
            _ => todo!(),
        }
    }

    fn detect_type_tuple(&self, angle: &AngleBracketedGenericArguments) -> DetectedType {
        assert!(angle.args.len() == 3);

        let shape = if let GenericArgument::Type(shape) = &angle.args[0] {
            self.detect_type_shape(shape)
        } else {
            todo!()
        };
        let kind = if let GenericArgument::Type(kind) = &angle.args[1] {
            self.detect_type_kind(kind)
        } else {
            todo!()
        };
        let device = if let GenericArgument::Type(device) = &angle.args[2] {
            self.detect_type_device(device)
        } else {
            todo!()
        };
        DetectedType::Declared(TensorType::new(shape, kind, device))
    }

    fn detect_type_shape(&self, type_: &Type) -> Vec<Dim> {
        match type_ {
            Type::Tuple(tup) => tup
                .elems
                .iter()
                .map(|item| self.detect_type_dim(item))
                .collect(),
            _ => todo!("This type_ is not handled in detect_type_shape"),
        }
    }

    fn detect_type_dim(&self, type_: &Type) -> Dim {
        match type_ {
            Type::Path(path) => Dim::Symbol(path.path.segments[0].ident.to_string()),
            Type::TraitObject(object) => {
                let bound = &object.bounds[0];
                match bound {
                    TypeParamBound::Trait(traits) => {
                        Dim::Symbol(traits.path.segments[0].ident.to_string())
                    }
                    _ => todo!("Finish bound in dim"),
                }
            }
            s => todo!("Finish dim! {s:?}"),
        }
    }

    fn detect_type_kind(&self, type_: &Type) -> Kind {
        match type_ {
            Type::Path(path) => Kind::Symbol(path.path.segments[0].ident.to_string()),
            _ => todo!("Finish type kind!"),
        }
    }

    fn detect_type_device(&self, type_: &Type) -> Device {
        match type_ {
            Type::Path(path) => Device::Symbol(path.path.segments[0].ident.to_string()),
            _ => todo!("Finish type device!"),
        }
    }

    fn detect_type_return(&self, return_type: ReturnType) -> (Option<Type>, DetectedType) {
        match return_type {
            ReturnType::Type(_, ty) => {
                let mut ty = ty.clone();
                let detected_type = self.detect_type_type(&mut *ty);
                (Some(*ty), detected_type)
            }
            ReturnType::Default => (None, DetectedType::NotTensor),
        }
    }

    fn assign_type(&mut self, ident: &Ident, type_: TensorType) {
        println!("Assigning {ident:?} {type_:?}");
        self.vars
            .insert(ident.clone(), DetectedType::Inferred(type_));
    }

    fn assign_return_type(&mut self, type_: TensorType) {
        self.return_type = DetectedType::Inferred(type_);
    }
}
fn compatible(left: &DetectedType, right: &DetectedType) -> bool {
    left == right
    // TODO
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Dim {
    Value(usize),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    Float,
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Device {
    Cpu,
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorType {
    shape: Vec<Dim>,
    kind: Kind,
    device: Device,
}

impl TensorType {
    pub fn new(shape: Vec<Dim>, kind: Kind, device: Device) -> Self {
        Self {
            shape,
            kind,
            device,
        }
    }
}

/// The `Fold` trait is a way to traverse an owned syntax tree and replace some
/// of its nodes.
///
/// Syn provides two other syntax tree traversal traits: `Visit` which walks a
/// shared borrow of a syntax tree, and `VisitMut` which walks an exclusive
/// borrow of a syntax tree and can mutate it in place.
///
/// All three traits have a method corresponding to each type of node in Syn's
/// syntax tree. All of these methods have default no-op implementations that
/// simply recurse on any child nodes. We can override only those methods for
/// which we want non-default behavior. In this case the traversal needs to
/// transform `Expr` and `Stmt` nodes.
impl Fold for Args {
    fn fold_fn_arg(&mut self, s: FnArg) -> FnArg {
        let (name, actual_type, detected_type): (Ident, Type, DetectedType) =
            self.detect_type_arg(s);
        if let DetectedType::Declared(tensor_type) = detected_type {
            self.assign_type(&name, tensor_type);
        }
        parse_quote!(#name: #actual_type)
    }

    fn fold_return_type(&mut self, s: ReturnType) -> ReturnType {
        let (actual_type, detected_type) = self.detect_type_return(s);
        if let DetectedType::Declared(tensor_type) = detected_type {
            self.assign_return_type(tensor_type);
        }
        if let Some(actual_type) = actual_type {
            parse_quote!(-> #actual_type)
        } else {
            parse_quote!()
        }
    }

    fn fold_stmt(&mut self, s: Stmt) -> Stmt {
        match s {
            Stmt::Local(s) => {
                if s.init.is_some() {
                    self.let_and_print(s)
                } else {
                    Stmt::Local(fold::fold_local(self, s))
                }
            }
            Stmt::Expr(ref e) => {
                // TODO also handle actual `return r;`.
                match e {
                    Expr::Call(call) => {
                        println!("{call:#?}");
                        let detected_type = self.detect_type_call(call);
                        if detected_type != self.return_type {
                            e.span().unwrap().error(format!("The return type \"{detected_type:?}\" does not match the expected return type \"{:?}\"", self.return_type)).emit();
                        }
                    }
                    _ => (),
                }
                fold::fold_stmt(self, s)
            }
            _ => fold::fold_stmt(self, s),
        }
    }
}

#[proc_macro_attribute]
pub fn tensor_check_fn(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemFn);

    // Parse the list of variables the user wanted to print.
    let mut args = Args::new();

    // Use a syntax tree traversal to transform the function body.
    let output = args.fold_item_fn(input);

    // Hand the resulting function body back to the compiler.
    TokenStream::from(quote!(#output))
}
