#![feature(proc_macro_diagnostic)]
mod tensor_type;
use crate::tensor_type::{Device, Dim, Kind, TensorType};
use log::debug;
use proc_macro::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::fold::{self, Fold};
use syn::spanned::Spanned;
use syn::UnOp::Neg;
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
    Shape(Vec<Dim>),
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
        let Local { mut pat, init, .. } = local;
        let expr = *init.unwrap().1;

        let ident = match &mut pat {
            Pat::Ident(ref p) => &p.ident,
            Pat::Type(ref mut t) => {
                let pat = &*t.pat;
                if let Pat::Ident(ref p) = pat {
                    // TODO Actually read the type of the tensor and affect it
                    let name = &p.ident;
                    let pat_type = self.detect_type_type(&mut t.ty);
                    if let DetectedType::Inferred(t) = pat_type {
                        self.assign_type(name, DetectedType::Inferred(t));
                    }
                    &p.ident
                } else {
                    todo!("let and print Type {t:?}");
                }
            }
            n => todo!("let and print {n:?}"),
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
            Expr::MethodCall(m) => {
                let receiver_type = self.get_type(&*m.receiver);
                match receiver_type {
                    DetectedType::Shape(shape) => todo!("Shape {shape:?}"),
                    DetectedType::NotTensor | DetectedType::NotDetected => (),
                    DetectedType::Inferred(type_) | DetectedType::Declared(type_) => {
                        let arg_types: Vec<_> =
                            m.args.iter().map(|item| self.get_type(item)).collect();
                        match &m.method.to_string()[..] {
                            "addmm" => {
                                // Assert shapes
                                if type_.shape.len() != 2 {
                                    m.receiver
                                        .span()
                                        .unwrap()
                                        .error(format!(
                                            "This tensor must be 2 dimensional but found {type_:?}"
                                        ))
                                        .emit();
                                }
                                assert!(arg_types.len() == 2);
                                let arg_types: Vec<_> = arg_types
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, arg_type)| match arg_type {
                                        DetectedType::Inferred(type_)
                                        | DetectedType::Declared(type_) => {
                                            if type_.shape.len() != 2 {
                                                m.args[i]
                                                    .span()
                                                    .unwrap()
                                                    .error(format!(
                                            "This tensor must be 2 dimensional but found {type_:?}"
                                        ))
                                                    .emit();
                                            }
                                            type_
                                        }
                                        _ => unreachable!(),
                                    })
                                    .collect();
                                // (U1, N) (B, M) (M, N)
                                // TODO check bias dim is 1
                                // for later because 1 vs U1 (should ideally be
                                // solved directly in the parser
                                let _u1_bias = &type_.shape[0];

                                let n_bias = &type_.shape[1];

                                let b_arg = &arg_types[0].shape[0];
                                let m_arg = &arg_types[0].shape[1];

                                let m_weight = &arg_types[1].shape[0];
                                let n_weight = &arg_types[1].shape[1];

                                if m_arg != m_weight {
                                    m.args[0]
                                        .span()
                                                    .unwrap()
                                                    .error(format!(
                                            "Second dim was expected to be {m_weight:?} but found {m_arg:?}")).emit();
                                }
                                if n_weight != n_bias {
                                    m.args[1]
                                        .span()
                                                    .unwrap()
                                                    .error(format!(
                                            "Second dim was expected to be {n_bias:?} but found {n_weight:?}")).emit();
                                }

                                // Assert kinds are identical
                                arg_types.iter().enumerate().for_each(|(i, arg_type)| {
                                    if type_.kind != arg_type.kind {
                                        m.args[i]
                                            .span()
                                            .unwrap()
                                            .error(format!(
                                                "Expected kind {:?} but found {:?}",
                                                type_.kind, arg_types[0].kind
                                            ))
                                            .emit();
                                    }
                                });

                                // Assert devices are identical
                                arg_types.iter().enumerate().for_each(|(i, arg_type)| {
                                    if type_.device != arg_type.device {
                                        m.args[i]
                                            .span()
                                            .unwrap()
                                            .error(format!(
                                                "Expected device {:?} but found {:?}",
                                                type_.device, arg_types[0].device
                                            ))
                                            .emit();
                                    }
                                });

                                let outtype = TensorType::new(
                                    vec![b_arg.clone(), n_weight.clone()],
                                    type_.kind,
                                    type_.device,
                                );
                                self.assign_type(&ident, DetectedType::Inferred(outtype));
                            }
                            "view" => {
                                assert!(m.args.len() == 1);
                                let outtype = self.detect_view_shape(&type_, &m.args[0]);
                                self.assign_type(&ident, DetectedType::Inferred(outtype));
                            }
                            "size" => {
                                let shape = type_.shape;
                                let outtype = DetectedType::Shape(shape);
                                self.assign_type(&ident, outtype);
                            }
                            m => todo!("Implement tensor method call {m:?}",),
                        }
                    }
                }
            }
            s => todo!("Other {s:?}"),
        }
        let init = self.fold_expr(expr);
        parse_quote! {
            let #pat =#init;
        }
    }

    fn detect_view_shape(&self, type_: &TensorType, view_arg: &Expr) -> TensorType {
        if let Expr::Tuple(tuple) = view_arg {
            // TODO fuse into 1 item with Option ?
            let mut filler = Dim::Mul(type_.shape.iter().map(|d| d.clone()).collect());
            let mut filler_used = false;
            let newshape: Vec<Option<Dim>> = tuple
                .elems
                .iter()
                .map(|elem| -> Option<Dim> {
                    match elem{
                        Expr::Unary(unary) => {
                            match (unary.op, &*unary.expr){
                                (Neg(_), Expr::Lit(expr)) =>match &expr.lit {
                     Lit::Int(int) => if int.base10_digits() == "1"{
                                        if filler_used{
                                            elem.span().unwrap().error("-1 already used, it can be used only once per view call");
                                        }
                                        filler_used= true;
                                        None
                    }else{
                        todo!("Else ! ")
                    },
                    _ => panic!("Can't handle non int cat"),
                                        },
                                        n => todo!("Detect view shape {n:?}")
                                    }
                                }
                        Expr::Index(index) => {
                            let tensor = self.get_type(&index.expr);
                            match (tensor, &*index.index){
                                (DetectedType::Shape(shape), Expr::Lit(expr_lit)) => {
                                    match &expr_lit.lit{
                                        Lit::Int(int) => {
                                    let int: usize = int.base10_parse().unwrap();
                                    if int >= shape.len(){
                                        int.span().unwrap().error("This index is invalid for this shape").emit();
                                    }
                                    let dim = shape[int].clone();
                                    filler.divide_by(dim.clone());
                                    Some(dim)
                                        },
                                        n => todo!("detect view shape lit {n:?}" )
                                        }

                                }
                                n => todo!("detect view shape AA {n:?}")
                            }
                        }
                                n => todo!("Detect view shape expr {n:?}")
                    }
                })
            .collect();
            filler.simplify();
            let newshape: Vec<Dim> = newshape
                .into_iter()
                .map(|elem| {
                    if let Some(elem) = elem {
                        elem
                    } else {
                        filler.clone()
                    }
                })
                .collect();

            let newtype = TensorType::new(newshape, type_.kind.clone(), type_.device.clone());
            if !newtype.is_view_compatible(type_) {
                view_arg
                    .span()
                    .unwrap()
                    .error(format!(
                        "This view {:?} is incompatible with tensor shape {:?}",
                        newtype.shape, type_.shape
                    ))
                    .emit();
            }
            newtype
        } else {
            todo!("detect view shape {view_arg:?}")
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
                    todo!("Too many path segments")
                }
            }
            Expr::Binary(binary) => {
                let left = self.get_type(&*binary.left);
                let right = self.get_type(&*binary.right);
                match (left, right) {
                    (DetectedType::NotTensor, DetectedType::NotTensor) => DetectedType::NotTensor,
                    (n, DetectedType::NotTensor) => n,
                    (DetectedType::NotTensor, n) => n,
                    (n, m) => {
                        if n != m {
                            binary
                                .span()
                                .unwrap()
                                .error(format!("Type are incompatible {n:?} and {m:?}"))
                                .emit();

                            DetectedType::NotDetected
                        } else {
                            m
                        }
                    }
                }
            }
            Expr::Lit(_) => DetectedType::NotTensor,
            Expr::Paren(p) => self.get_type(&*p.expr),
            Expr::Reference(reference) => self.get_type(&*reference.expr),
            Expr::Tuple(_tuple) => DetectedType::NotDetected,
            m => todo!("Expr {m:?}"),
        }
    }

    fn maybe_assign_type(&mut self, ident: &Ident, call: &ExprCall) {
        let detected_type = self.detect_type_call(call);
        if let DetectedType::Inferred(type_) = detected_type {
            self.assign_type(&ident, DetectedType::Inferred(type_));
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
            expr => todo!("type call {expr:?}"),
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

        let type_ = types
            .into_iter()
            .reduce(|mut accum, item| {
                assert!(item.kind == accum.kind);
                assert!(item.device == accum.device);
                assert!(item.shape.len() == accum.shape.len());
                for (i, (dim_acc, dim_it)) in
                    accum.shape.iter_mut().zip(item.shape.iter()).enumerate()
                {
                    let i = i as i64;
                    if i != dim {
                        assert!(dim_acc == dim_it);
                    } else {
                        if let Dim::Add(a) = dim_acc {
                            a.push(dim_it.clone());
                        } else {
                            *dim_acc = Dim::Add(vec![dim_acc.clone(), dim_it.clone()]);
                        }
                    }
                }
                accum
            })
            .unwrap();

        type_
    }

    fn detect_type(&self, expr: &Expr) -> TensorType {
        match expr {
            Expr::Path(path) => {
                assert!(path.path.segments.len() == 1);
                let name = &path.path.segments[0].ident;
                match self.vars.get(name) {
                    Some(DetectedType::Declared(tensor_type)) => tensor_type.clone(),
                    Some(DetectedType::Inferred(tensor_type)) => tensor_type.clone(),
                    n => {
                        name.span()
                            .unwrap()
                            .error("Couldn't get the tensor type for this variable")
                            .emit();
                        todo!("the name {name:?} was resolved as {n:?}");
                    }
                }
            }
            expr => todo!("Expr detect type {expr:?}"),
        }
    }

    fn detect_shape(&self, expr: &Expr) -> Vec<Dim> {
        match expr {
            Expr::Reference(expr_ref) => match &*expr_ref.expr {
                Expr::Array(ExprArray { elems, .. }) => elems
                    .iter()
                    .filter_map(|elem| match elem {
                        Expr::Lit(expr_lit) => match &expr_lit.lit {
                            Lit::Int(lit_int) => {
                                Some(Dim::from_num(lit_int.base10_parse().unwrap()))
                            }
                            lit => todo!("Detect shape lit {lit:?}"),
                        },
                        _ => None,
                    })
                    .collect(),
                expr => todo!("Detect shape2 {expr:?}"),
            },
            expr => todo!("Detect shape {expr:?}"),
        }
    }

    fn detect_kind_device(&self, expr: &Expr) -> (Kind, Device) {
        match expr {
            Expr::Tuple(ExprTuple { elems, .. }) => {
                assert!(elems.len() == 2);
                let kind = match &elems[0] {
                    Expr::Path(ExprPath { path, .. }) => {
                        let name = &path.segments[1].ident.to_string();
                        if name == "Float" {
                            Kind::Float
                        } else {
                            todo!("Detect kind device kind literal {name:?}")
                        }
                    }
                    expr => todo!("detect kind device kind {expr:?}"),
                };
                let device = match &elems[1] {
                    Expr::Path(ExprPath { path, .. }) => {
                        let device = &path.segments[1].ident.to_string();
                        if device == "Cpu" {
                            Device::Cpu
                        } else {
                            todo!("Detect kind device device literal {device:?}")
                        }
                    }
                    expr => todo!("detect kind device device {expr:?}"),
                };
                (kind, device)
            }
            expr => todo!("Detect kind device {expr:?}"),
        }
    }

    fn detect_type_arg(&self, arg: FnArg) -> (Ident, Type, DetectedType) {
        match arg {
            FnArg::Typed(PatType { pat, ty, .. }) => {
                let name = match *pat {
                    Pat::Ident(ident) => ident.ident,
                    pat => todo!("Pat {pat:?}"),
                };
                let mut ty = ty.clone();
                let detected_type = self.detect_type_type(&mut *ty);
                (name, *ty, detected_type)
            }
            fn_arg => todo!("FnArg {fn_arg:?}"),
        }
    }

    fn detect_type_type(&self, mut ty: &mut Type) -> DetectedType {
        if let Type::Reference(ref_) = ty {
            ty = &mut *ref_.elem;
        }
        match ty {
            Type::Path(TypePath { ref mut path, .. }) => {
                let mut segment = &mut path.segments[0];
                if segment.ident.to_string() == "Tensor" {
                    let detected_type = match &mut segment.arguments {
                        PathArguments::AngleBracketed(angle) => self.detect_type_tuple(angle),
                        PathArguments::None => DetectedType::NotDetected,
                        n => todo!("Path arguments {n:?}"),
                    };
                    segment.arguments = PathArguments::None;
                    detected_type
                } else {
                    DetectedType::NotTensor
                }
            }
            ty => todo!("Type {ty:?} not handled yet."),
        }
    }

    fn detect_type_tuple(&self, angle: &AngleBracketedGenericArguments) -> DetectedType {
        assert!(angle.args.len() >= 1);

        let shape = if let GenericArgument::Type(shape) = &angle.args[0] {
            self.detect_type_shape(shape)
        } else {
            todo!("Could not detect shape")
        };

        let kind = if angle.args.len() > 1 {
            if let GenericArgument::Type(kind) = &angle.args[1] {
                self.detect_type_kind(kind)
            } else {
                todo!("Could not detect kind")
            }
        } else {
            Kind::Implicit
        };
        let device = if angle.args.len() > 2 {
            if let GenericArgument::Type(device) = &angle.args[2] {
                self.detect_type_device(device)
            } else {
                todo!("Could not detect device")
            }
        } else {
            Device::Implicit
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
            Type::Path(path) => Dim::from_symbol(path.path.segments[0].ident.to_string()),
            Type::TraitObject(object) => {
                let dims: Vec<Dim> = object
                    .bounds
                    .iter()
                    .map(|bound| self.detect_type_bound(bound))
                    .collect();
                Dim::from_add(dims)
            }
            s => todo!("Finish dim! {s:?}"),
        }
    }

    fn detect_type_bound(&self, bound: &TypeParamBound) -> Dim {
        match bound {
            TypeParamBound::Trait(traits) => Dim::Symbol(traits.path.segments[0].ident.to_string()),
            _ => todo!("Finish bound in dim"),
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

    fn assign_type(&mut self, ident: &Ident, type_: DetectedType) {
        debug!("Assigning shape {:?} {type_:?}", ident.to_string());
        self.vars.insert(ident.clone(), type_);
    }

    fn assign_return_type(&mut self, type_: TensorType) {
        self.return_type = DetectedType::Inferred(type_);
    }
}
fn compatible(left: &DetectedType, right: &DetectedType) -> bool {
    left == right
    // TODO
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
            self.assign_type(&name, DetectedType::Declared(tensor_type));
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
                        let detected_type = self.detect_type_call(call);
                        if detected_type != self.return_type {
                            e.span().unwrap().error(format!("The return type \"{detected_type:?}\" does not match the expected return type \"{:?}\"", self.return_type)).emit();
                        } else {
                            debug!(
                                "There is a correct match between {detected_type:?} and {:?}",
                                self.return_type
                            );
                        }
                    }
                    Expr::MethodCall(_call) => {
                        todo!("Handle method calls");
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
