use crate::function::{Args, Signature};
use crate::tensor_type::TensorType;
use crate::tensor_type_detection::{detect_type_type, DetectedType};
use crate::Module;
use std::collections::HashMap;
use syn::{fold::Fold, Ident, ImplItem, ItemFn, ItemImpl, ItemStruct, Type};

fn resolve(type_: &Type) -> &Ident {
    match type_ {
        Type::Path(path) => &path.path.segments[0].ident,
        n => todo!("Resolve {n:?}"),
    }
}

#[derive(Clone)]
pub(crate) struct Struct {
    pub(crate) members: HashMap<Ident, TensorType>,
    pub(crate) methods: HashMap<Ident, Signature>,
}

impl Default for Struct {
    fn default() -> Self {
        Self {
            members: HashMap::new(),
            methods: HashMap::new(),
        }
    }
}

impl Module {
    pub fn inspect_fns(&mut self, function: &mut ItemFn) {
        // Parse the list of variables the user wanted to print.
        let mut args = Args::new(self.clone());
        // use a syntax tree traversal to transform the function body.
        let output = args.fold_item_fn(function.clone());
        let name = function.sig.ident.clone();
        self.fns.insert(name, args.signature());
        *function = output
    }
    pub fn inspect_members(&mut self, struct_: &mut ItemStruct) {
        let tensor_fields: HashMap<Ident, TensorType> = struct_
            .fields
            .iter_mut()
            .flat_map(|field| {
                if let (Some(ident), DetectedType::Declared(tensor_type)) =
                    (&field.ident, detect_type_type(&mut field.ty))
                {
                    Some((ident.clone(), tensor_type))
                } else {
                    None
                }
            })
            .collect();
        if !tensor_fields.is_empty() {
            let mut new_struct_ = Struct::default();
            new_struct_.members = tensor_fields;
            self.structs.insert(struct_.ident.clone(), new_struct_);
        }
    }

    pub fn inspect_methods(&mut self, impl_: &mut ItemImpl) {
        let module = self.clone();

        let name = resolve(&impl_.self_ty);
        if let Some(struct_) = self.structs.get_mut(&name) {
            for item in &mut impl_.items {
                if let ImplItem::Method(method) = item {
                    let module: Module = module.clone();
                    let mut args = Args::new(module);
                    args.set_self(name.clone());
                    // use a syntax tree traversal to transform the function body.
                    let output = args.fold_impl_item_method(method.clone());
                    let name = method.sig.ident.clone();
                    let signature = args.signature();
                    struct_.methods.insert(name, signature);
                    *method = output
                }
            }
        }
    }
}
