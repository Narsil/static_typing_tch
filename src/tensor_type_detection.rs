use crate::tensor_type::{Device, Dim, Kind, TensorType};
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
            n => todo!("Handle {n:?}"),
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

fn detect_type_tuple(angle: &AngleBracketedGenericArguments) -> DetectedType {
    assert!(angle.args.len() >= 1);

    let shape = if let GenericArgument::Type(shape) = &angle.args[0] {
        detect_type_shape(shape)
    } else {
        todo!("Could not detect shape")
    };

    let kind = if angle.args.len() > 1 {
        if let GenericArgument::Type(kind) = &angle.args[1] {
            detect_type_kind(kind)
        } else {
            todo!("Could not detect kind")
        }
    } else {
        Kind::Implicit
    };
    let device = if angle.args.len() > 2 {
        if let GenericArgument::Type(device) = &angle.args[2] {
            detect_type_device(device)
        } else {
            todo!("Could not detect device")
        }
    } else {
        Device::Implicit
    };
    DetectedType::Declared(TensorType::new(shape, kind, device))
}

fn detect_type_shape(type_: &Type) -> Vec<Dim> {
    match type_ {
        Type::Tuple(tup) => tup.elems.iter().map(|item| detect_type_dim(item)).collect(),
        _ => todo!("This type_ is not handled in detect_type_shape"),
    }
}

fn detect_type_dim(type_: &Type) -> Dim {
    match type_ {
        Type::Path(path) => Dim::from_symbol(path.path.segments[0].ident.to_string()),
        Type::TraitObject(object) => {
            let dims: Vec<Dim> = object
                .bounds
                .iter()
                .map(|bound| detect_type_bound(bound))
                .collect();
            Dim::from_add(dims)
        }
        s => todo!("Finish dim! {s:?}"),
    }
}

fn detect_type_bound(bound: &TypeParamBound) -> Dim {
    match bound {
        TypeParamBound::Trait(traits) => Dim::Symbol(traits.path.segments[0].ident.to_string()),
        _ => todo!("Finish bound in dim"),
    }
}

fn detect_type_kind(type_: &Type) -> Kind {
    match type_ {
        Type::Path(path) => Kind::Symbol(path.path.segments[0].ident.to_string()),
        _ => todo!("Finish type kind!"),
    }
}

fn detect_type_device(type_: &Type) -> Device {
    match type_ {
        Type::Path(path) => Device::Symbol(path.path.segments[0].ident.to_string()),
        _ => todo!("Finish type device!"),
    }
}
