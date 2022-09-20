use crate::tensor_type::{Device, Dim, Kind, TensorType};
use crate::tensor_type_detection::{detect_type_type, DetectedType};
use crate::Module;
use log::debug;
use std::collections::HashMap;
use syn::fold::{self, Fold};
use syn::spanned::Spanned;
use syn::UnOp::Neg;
use syn::{
    parse_quote, BinOp, Expr, ExprArray, ExprCall, ExprMethodCall, ExprPath, ExprTuple, FnArg,
    Ident, Lit, Local, Member, Pat, PatType, ReturnType, Stmt, Type,
};

pub(crate) struct Args {
    // TODO This could be a reference
    module: Module,
    current_self: Option<Ident>,
    vars: HashMap<Ident, DetectedType>,
    args: Vec<DetectedType>,
    return_type: DetectedType,
}

impl Args {
    pub fn new(module: Module) -> Self {
        Self {
            vars: HashMap::new(),
            return_type: DetectedType::NotDetected,
            args: vec![],
            current_self: None,
            module,
        }
    }

    pub(crate) fn set_self(&mut self, name: Ident) {
        self.current_self = Some(name);
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Signature {
    args: Vec<DetectedType>,
    return_type: DetectedType,
}

impl Args {
    pub fn signature(self) -> Signature {
        Signature {
            args: self.args,
            return_type: self.return_type,
        }
    }
    fn let_and_print(&mut self, local: Local) -> Stmt {
        let Local { mut pat, init, .. } = local;
        let expr = *init.unwrap().1;

        let ident = match &mut pat {
            Pat::Ident(ref p) => &p.ident,
            Pat::Type(ref mut t) => {
                let pat = &*t.pat;
                if let Pat::Ident(ref p) = pat {
                    let name = &p.ident;
                    let pat_type = detect_type_type(&mut t.ty);
                    if let DetectedType::Inferred(t) = pat_type {
                        self.assign_type(name, DetectedType::Inferred(t));
                    }
                    &p.ident
                } else {
                    t.span().unwrap().error("let and print Type").emit();
                    let init = self.fold_expr(expr);
                    let stmt: Stmt = parse_quote! {
                        let #pat = #init
                    };
                    return stmt;
                }
            }
            n => {
                n.span().unwrap().error("let and print").emit();
                let init = self.fold_expr(expr);
                let stmt: Stmt = parse_quote! {
                    let #pat = #init
                };
                return stmt;
            }
        };
        match &expr {
            Expr::Call(m) => {
                if let Expr::Path(p) = &*m.func {
                    let ident = &p.path.segments[0].ident;
                    if let Some(sig) = self.module.fns.get(ident) {
                        self.check_valid_call(&m, sig);
                    }
                }
                self.maybe_assign_type(&ident, m)
            }
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
                    DetectedType::Shape(_) => {
                        m.receiver.span().unwrap().error("Shape").emit();
                        //TODO
                    }
                    DetectedType::NotTensor | DetectedType::NotDetected => (),
                    DetectedType::Inferred(type_) | DetectedType::Declared(type_) => {
                        let arg_types: Vec<_> =
                            m.args.iter().map(|item| self.get_type(item)).collect();
                        match &m.method.to_string()[..] {
                            "addmm" => {
                                // Assert shapes
                                if type_.shape.len() != 1 {
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
                                let n_bias = &type_.shape[0];

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
                                if let Some(outtype) = self.detect_view_shape(&type_, &m.args[0]) {
                                    self.assign_type(&ident, DetectedType::Inferred(outtype));
                                }
                            }
                            "size" => {
                                let shape = type_.shape;
                                let outtype = DetectedType::Shape(shape);
                                self.assign_type(&ident, outtype);
                            }
                            m => {
                                m.span()
                                    .unwrap()
                                    .error(format!("Implement tensor method call {m:?}",))
                                    .emit();
                            }
                        }
                    }
                }
            }
            s => {
                s.span().unwrap().warning(format!("Unhandled")).emit();
            }
        }
        let init = self.fold_expr(expr);
        parse_quote! {
            let #pat =#init;
        }
    }

    fn get_signature_arg(&self, expr: &Expr) -> DetectedType {
        match expr {
            Expr::Reference(reference) => self.get_signature_arg(&*reference.expr),
            Expr::Path(path) => {
                assert!(path.path.segments.len() == 1);
                self.vars
                    .get(&path.path.segments[0].ident)
                    .unwrap_or(&DetectedType::NotDetected)
                    .clone()
            }
            n => {
                n.span().unwrap().error("get_signature_arg").emit();
                DetectedType::NotDetected
            }
        }
    }

    fn check_valid_call(&self, expr_call: &ExprCall, sig: &Signature) {
        let args = &sig.args;

        args.iter()
            .zip(expr_call.args.iter())
            .for_each(|(arg, current_arg)| {
                let arg_type = self.get_signature_arg(current_arg);
                if !arg_type.is_arg_compatible(arg) {
                    current_arg
                        .span()
                        .unwrap()
                        .error(format!(
                            "This call is invalid, expected {arg:#?} but got {arg_type:#?}"
                        ))
                        .emit();
                }
            })
    }

    fn detect_view_shape(&self, type_: &TensorType, view_arg: &Expr) -> Option<TensorType> {
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

                    elem.span().unwrap().error("Unhandled detect view shape").emit();
                    None

                    },
                    n => {
                        n.span().unwrap().error("Can't handle non int cat").emit();
                        None
                    },
                                        },
                                (_, n) => {
                        n.span().unwrap().error("Detect view shape").emit();
                        None
                    },
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
                                        n => {
                                            n.span().unwrap().error("detect view shape lit").emit();
                                            None
                                        }
                                        }

                                }
                                (_, n) => {
                                            n.span().unwrap().error("detect view shape AA").emit();
                                            None
                                }
                            }
                        }
                                n => {
                                            n.span().unwrap().error("detect view expr AA").emit();
                                            None
                                }
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
            Some(newtype)
        } else {
            view_arg
                .span()
                .unwrap()
                .error(format!("detect view shape {view_arg:?}"))
                .emit();
            None
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
                    path.span().unwrap().error("Too many path segments").emit();
                    DetectedType::NotDetected
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
            // // TODO
            Expr::Call(_tuple) => DetectedType::NotTensor,
            Expr::Field(field) => {
                if let Some(current_self) = &self.current_self {
                    if let Some(struct_) = self.module.structs.get(current_self) {
                        if let Member::Named(name) = &field.member {
                            if let Some(type_) = struct_.members.get(name) {
                                return DetectedType::Inferred(type_.clone());
                            }
                        }
                    }
                }
                field.span().unwrap().error("Expr Field").emit();
                DetectedType::NotTensor
            }
            Expr::MethodCall(expr) => self.detect_type_method_call(expr),
            m => {
                m.span().unwrap().warning("Unhandled Expr").emit();
                DetectedType::NotDetected
            }
        }
    }

    fn maybe_assign_type(&mut self, ident: &Ident, call: &ExprCall) {
        let detected_type = self.detect_type_call(call);
        if let DetectedType::Inferred(type_) = detected_type {
            self.assign_type(&ident, DetectedType::Inferred(type_));
        }
    }

    fn detect_type_method_call(&self, call: &ExprMethodCall) -> DetectedType {
        let type_ = self.get_type(&call.receiver);
        match &type_ {
            DetectedType::Inferred(x) | DetectedType::Declared(x) => {
                match &call.method.to_string()[..] {
                    "linear" => {
                        assert!(call.args.len() == 2);
                        let weight_type = self.detect_type(&call.args[0]);
                        let bias = self.detect_type(&call.args[1]);

                        if x.kind != weight_type.kind {
                            call.args[0]
                                .span()
                                .unwrap()
                                .error("Mismatched types expected {x:?} but found {weight_type:?}")
                                .emit();
                        }
                        if bias.kind != weight_type.kind {
                            call.args[1]
                                .span()
                                .unwrap()
                                .error(
                                    "Mismatched types expected {weight_type:?} but found {bias:?}",
                                )
                                .emit();
                        }
                        if x.device != weight_type.device {
                            call.args[0]
                                .span()
                                .unwrap()
                                .error("Mismatched types expected {x:?} but found {weight_type:?}")
                                .emit();
                        }
                        if bias.kind != weight_type.kind {
                            call.args[1]
                                .span()
                                .unwrap()
                                .error(
                                    "Mismatched types expected {weight_type:?} but found {bias:?}",
                                )
                                .emit();
                        }
                        let outshape = vec![x.shape[0].clone(), weight_type.shape[0].clone()];
                        DetectedType::Inferred(TensorType::new(
                            outshape,
                            x.kind.clone(),
                            x.device.clone(),
                        ))
                    }
                    "addmm" => {
                        assert!(call.args.len() == 2);
                        let bias = x;
                        let x = self.detect_type(&call.args[0]);
                        let weight_type = self.detect_type(&call.args[1]);

                        if x.kind != weight_type.kind {
                            call.args[0]
                                .span()
                                .unwrap()
                                .error("Mismatched types expected {x:?} but found {weight_type:?}")
                                .emit();
                        }
                        if bias.kind != weight_type.kind {
                            call.args[1]
                                .span()
                                .unwrap()
                                .error(
                                    "Mismatched types expected {weight_type:?} but found {bias:?}",
                                )
                                .emit();
                        }
                        if x.device != weight_type.device {
                            call.args[0]
                                .span()
                                .unwrap()
                                .error("Mismatched types expected {x:?} but found {weight_type:?}")
                                .emit();
                        }
                        if bias.kind != weight_type.kind {
                            call.args[1]
                                .span()
                                .unwrap()
                                .error(
                                    "Mismatched types expected {weight_type:?} but found {bias:?}",
                                )
                                .emit();
                        }
                        let outshape = vec![x.shape[0].clone(), weight_type.shape[1].clone()];
                        DetectedType::Inferred(TensorType::new(
                            outshape,
                            x.kind.clone(),
                            x.device.clone(),
                        ))
                    }
                    "view" => {
                        if let Some(outtype) = self.detect_view_shape(&x, &call.args[0]) {
                            DetectedType::Inferred(outtype)
                        } else {
                            // Warning already raised by fn
                            DetectedType::NotDetected
                        }
                    }
                    n => {
                        n.span().unwrap().error("Method call").emit();
                        DetectedType::NotDetected
                    }
                }
            }
            _ => {
                call.receiver
                    .span()
                    .unwrap()
                    .error("Cannot get type method call")
                    .emit();
                DetectedType::NotDetected
            }
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
                            // args[0]
                            //     .span()
                            //     .unwrap()
                            //     .warning(format!("Shape {shape:?}"))
                            //     .emit();
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
            expr => {
                expr.span().unwrap().error("type call").emit();
                DetectedType::NotDetected
            }
        }
    }

    fn detect_cat_type(&self, tensors: &Expr, dim: &Expr) -> TensorType {
        let dim: i64 = match dim {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Int(int) => int.base10_parse().unwrap(),
                n => {
                    n.span().unwrap().error("Can't handle non int cat").emit();
                    0
                }
            },
            n => {
                n.span().unwrap().error("Can't handle non int cat").emit();
                0
            }
        };
        let types: Vec<TensorType> = match tensors {
            Expr::Reference(reference) => match &*reference.expr {
                Expr::Array(array) => array
                    .elems
                    .iter()
                    .map(|item| self.detect_type(item))
                    .collect(),
                n => {
                    n.span()
                        .unwrap()
                        .error("cat is not supposed to be call that way")
                        .emit();
                    vec![]
                }
            },
            n => {
                n.span()
                    .unwrap()
                    .error("cat is not supposed to be call that way")
                    .emit();
                vec![]
            }
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
        let empty = TensorType::new(vec![], Kind::Implicit, Device::Implicit);
        match expr {
            Expr::Reference(reference) => self.detect_type(&*reference.expr),
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
                        empty
                    }
                }
            }
            Expr::Field(field) => {
                if let Some(current_self) = &self.current_self {
                    if let Some(struct_) = self.module.structs.get(current_self) {
                        if let Member::Named(name) = &field.member {
                            if let Some(type_) = struct_.members.get(name) {
                                return type_.clone();
                            }
                        }
                    }
                }
                field
                    .span()
                    .unwrap()
                    .error("Couldn't get the tensor type for this variable")
                    .emit();
                empty
            }
            Expr::Call(call) => {
                assert!(call.func == parse_quote!(Some));
                self.detect_type(&call.args[0])
            }
            expr => {
                expr.span()
                    .unwrap()
                    .error("Expr couldn't detect type")
                    .emit();
                empty
            }
        }
    }
    fn detect_dim(&self, expr: &Expr) -> Dim {
        match expr {
            Expr::Lit(expr_lit) => match &expr_lit.lit {
                Lit::Int(lit_int) => Dim::from_num(lit_int.base10_parse().unwrap()),
                lit => {
                    lit.span().unwrap().error("Detect shape lit {lit:?}").emit();
                    Dim::from_num(0)
                }
            },
            Expr::Binary(expr_bin) => match expr_bin.op {
                BinOp::Mul(_) => {
                    let left = self.detect_dim(&*expr_bin.left);
                    let right = self.detect_dim(&*expr_bin.right);
                    left * right
                }
                n => {
                    n.span().unwrap().error("detect shape operator").emit();
                    Dim::from_num(0)
                }
            },
            n => {
                n.span().unwrap().error("detect shape Handle").emit();
                Dim::from_num(0)
            }
        }
    }

    fn detect_shape(&self, expr: &Expr) -> Vec<Dim> {
        match expr {
            Expr::Reference(expr_ref) => match &*expr_ref.expr {
                Expr::Array(ExprArray { elems, .. }) => {
                    elems.iter().map(|elem| self.detect_dim(elem)).collect()
                }
                expr => {
                    expr.span().unwrap().error("Detect shape2").emit();
                    vec![]
                }
            },
            expr => {
                expr.span().unwrap().error("Detect shape").emit();
                vec![]
            }
        }
    }

    fn detect_kind_device(&self, expr: &Expr) -> (Kind, Device) {
        match expr {
            Expr::Tuple(ExprTuple { elems, .. }) => {
                assert!(elems.len() == 2);
                let kind = match &elems[0] {
                    Expr::Path(ExprPath { path, .. }) => {
                        let name = &path.segments[1].ident;
                        if name.to_string() == "Float" {
                            Kind::Float
                        } else {
                            name.span()
                                .unwrap()
                                .error(format!(
                                    "Detect kind device device literal {:?}",
                                    name.to_string()
                                ))
                                .emit();
                            Kind::Implicit
                        }
                    }
                    expr => {
                        expr.span().unwrap().error("Detect kind device kind").emit();
                        Kind::Implicit
                    }
                };
                let device = match &elems[1] {
                    Expr::Path(ExprPath { path, .. }) => {
                        let device = &path.segments[1].ident;
                        if device.to_string() == "Cpu" {
                            Device::Cpu
                        } else {
                            device
                                .span()
                                .unwrap()
                                .error(format!(
                                    "Detect kind device device literal {:?}",
                                    device.to_string()
                                ))
                                .emit();
                            Device::Implicit
                        }
                    }
                    expr => {
                        expr.span()
                            .unwrap()
                            .error("Detect kind device device literal")
                            .emit();
                        Device::Implicit
                    }
                };
                (kind, device)
            }
            expr => {
                expr.span().unwrap().error("Detect kind device ").emit();
                (Kind::Implicit, Device::Implicit)
            }
        }
    }

    fn detect_type_arg(&self, arg: &mut FnArg) -> Option<(Ident, DetectedType)> {
        match arg {
            FnArg::Typed(PatType { pat, ty, .. }) => {
                let name = match &**pat {
                    Pat::Ident(ident) => &ident.ident,
                    pat => {
                        pat.span().unwrap().error("Unhandled pat").emit();
                        return None;
                    }
                };
                let mut ty = ty.clone();
                let detected_type = detect_type_type(&mut *ty);
                let actual_type = *ty;
                // Remove the typing
                let result = Some((name.clone(), detected_type));
                *arg = parse_quote!(#name: #actual_type);
                result
            }
            FnArg::Receiver(_) => None,
        }
    }

    fn detect_type_return(&self, return_type: ReturnType) -> (Option<Type>, DetectedType) {
        match return_type {
            ReturnType::Type(_, ty) => {
                let mut ty = ty.clone();
                let detected_type = detect_type_type(&mut *ty);
                (Some(*ty), detected_type)
            }
            ReturnType::Default => (None, DetectedType::NotTensor),
        }
    }

    fn assign_type(&mut self, ident: &Ident, type_: DetectedType) {
        debug!("Assigning shape {:?} {type_:?}", ident.to_string());
        self.vars.insert(ident.clone(), type_);
    }
}
fn compatible(left: &DetectedType, right: &DetectedType) -> bool {
    left == right
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
    fn fold_fn_arg(&mut self, mut s: FnArg) -> FnArg {
        if let Some((name, detected_type)) = self.detect_type_arg(&mut s) {
            self.args.push(detected_type.clone());
            if let DetectedType::Declared(tensor_type) = detected_type {
                self.assign_type(&name, DetectedType::Declared(tensor_type));
            }
        }
        s
    }

    fn fold_return_type(&mut self, s: ReturnType) -> ReturnType {
        let (actual_type, detected_type) = self.detect_type_return(s);
        if let DetectedType::Declared(type_) = detected_type {
            // modify this so that Eq actually validates
            // Inferred == Declared
            self.return_type = DetectedType::Inferred(type_);
        } else {
            self.return_type = detected_type;
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
                    Expr::MethodCall(call) => {
                        let detected_type = self.detect_type_method_call(call);
                        if detected_type != self.return_type {
                            e.span().unwrap().error(format!("The return type \"{detected_type:?}\" does not match the expected return type \"{:?}\"", self.return_type)).emit();
                        } else {
                            debug!(
                                "There is a correct match between {detected_type:?} and {:?}",
                                self.return_type
                            );
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
