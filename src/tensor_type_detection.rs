use crate::tensor_type::{Device, Dim, Kind, TensorType};
use syn::spanned::Spanned;
use syn::{
    AngleBracketedGenericArguments, GenericArgument, PathArguments, Type, TypeParamBound, TypePath,
};
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectedType {
    NotDetected,
    NotTensor,
    Shape(Vec<Dim>),
    Inferred(TensorType),
    Declared(TensorType),
}

impl DetectedType {
    pub fn is_arg_compatible(&self, other: &DetectedType) -> bool {
        match (self, other) {
            (
                DetectedType::Inferred(left) | DetectedType::Declared(left),
                DetectedType::Inferred(right) | DetectedType::Declared(right),
            ) => left.is_arg_compatible(right),
            // TODO maybe more cases to handle
            (n, m) => n == m,
        }
    }
}

pub fn detect_type_type(mut ty: &mut Type) -> DetectedType {
    if let Type::Reference(ref_) = ty {
        ty = &mut *ref_.elem;
    }
    match ty {
        Type::Path(TypePath { ref mut path, .. }) => {
            let mut segment = &mut path.segments[0];
            if segment.ident.to_string() == "Tensor" {
                let detected_type = match &mut segment.arguments {
                    PathArguments::AngleBracketed(angle) => detect_type_tuple(angle),
                    PathArguments::None => {
                        segment
                            .ident
                            .span()
                            .unwrap()
                            .error("Missing shape hint")
                            .emit();
                        DetectedType::NotDetected
                    }
                    _ => {
                        segment
                            .ident
                            .span()
                            .unwrap()
                            .error("Path arguments unhandled")
                            .emit();
                        DetectedType::NotDetected
                    }
                };
                segment.arguments = PathArguments::None;
                detected_type
            } else {
                DetectedType::NotTensor
            }
        }
        ty => {
            ty.span().unwrap().error("Type not handled yet.").emit();
            DetectedType::NotDetected
        }
    }
}

fn detect_type_tuple(angle: &AngleBracketedGenericArguments) -> DetectedType {
    assert!(angle.args.len() >= 1);

    let shape = if let GenericArgument::Type(shape) = &angle.args[0] {
        detect_type_shape(shape)
    } else {
        angle.args[0]
            .span()
            .unwrap()
            .error("Could not detect shape")
            .emit();
        vec![]
    };

    let kind = if angle.args.len() > 1 {
        if let GenericArgument::Type(kind) = &angle.args[1] {
            detect_type_kind(kind)
        } else {
            angle.args[1]
                .span()
                .unwrap()
                .error("Could not detect kind")
                .emit();
            Kind::Implicit
        }
    } else {
        Kind::Implicit
    };
    let device = if angle.args.len() > 2 {
        if let GenericArgument::Type(device) = &angle.args[2] {
            detect_type_device(device)
        } else {
            angle.args[2]
                .span()
                .unwrap()
                .error("Could not detect device")
                .emit();
            Device::Implicit
        }
    } else {
        Device::Implicit
    };
    DetectedType::Declared(TensorType::new(shape, kind, device))
}

fn detect_type_shape(type_: &Type) -> Vec<Dim> {
    match type_ {
        Type::Tuple(tup) => tup
            .elems
            .iter()
            .flat_map(|item| detect_type_dim(item))
            .collect(),
        _ => {
            type_
                .span()
                .unwrap()
                .error("This type_ is not handled in detect_type_shape")
                .emit();
            vec![]
        }
    }
}

fn detect_type_dim(type_: &Type) -> Option<Dim> {
    match type_ {
        Type::Path(path) => Some(Dim::from_symbol(path.path.segments[0].ident.to_string())),
        Type::TraitObject(object) => {
            let dims: Vec<Dim> = object
                .bounds
                .iter()
                .flat_map(|bound| detect_type_bound(bound))
                .collect();
            Some(Dim::from_add(dims))
        }
        s => {
            s.span()
                .unwrap()
                .error("Detecting dim is unfinihsed")
                .emit();
            None
        }
    }
}

fn detect_type_bound(bound: &TypeParamBound) -> Option<Dim> {
    match bound {
        TypeParamBound::Trait(traits) => {
            Some(Dim::Symbol(traits.path.segments[0].ident.to_string()))
        }
        n => {
            n.span().unwrap().error("Finish bound in dim").emit();
            None
        }
    }
}

fn detect_type_kind(type_: &Type) -> Kind {
    match type_ {
        Type::Path(path) => Kind::Symbol(path.path.segments[0].ident.to_string()),
        n => {
            n.span().unwrap().error("Finish type kind!").emit();
            Kind::Implicit
        }
    }
}

fn detect_type_device(type_: &Type) -> Device {
    match type_ {
        Type::Path(path) => Device::Symbol(path.path.segments[0].ident.to_string()),
        n => {
            n.span().unwrap().error("Finish type device!").emit();
            Device::Implicit
        }
    }
}
