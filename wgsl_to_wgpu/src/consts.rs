use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{MatrixVectorTypes, ModulePath, TypePath, wgsl::rust_type};

pub fn consts<F>(module: &naga::Module, demangle: F) -> Vec<(TypePath, TokenStream)>
where
    F: Fn(&str) -> TypePath,
{
    // Create matching Rust constants for WGSl constants.
    module
        .constants
        .iter()
        .filter_map(|(_, t)| {
            let path = demangle(t.name.as_ref()?);
            let name = Ident::new(&path.name, Span::call_site());

            let type_and_value = match &module.global_expressions[t.init] {
                naga::Expression::Literal(literal) => match literal {
                    naga::Literal::F64(v) => Some(quote!(f64 = #v)),
                    naga::Literal::F32(v) => Some(quote!(f32 = #v)),
                    naga::Literal::U32(v) => Some(quote!(u32 = #v)),
                    naga::Literal::I32(v) => Some(quote!(i32 = #v)),
                    naga::Literal::U64(v) => Some(quote!(u64 = #v)),
                    naga::Literal::Bool(v) => Some(quote!(bool = #v)),
                    naga::Literal::I64(v) => Some(quote!(i64 = #v)),
                    naga::Literal::AbstractInt(v) => Some(quote!(i64 = #v)),
                    naga::Literal::AbstractFloat(v) => Some(quote!(f64 = #v)),
                    naga::Literal::F16(v) => {
                        let v = v.to_f32();
                        Some(quote!(half::f16 = half::f16::from_f32_const(#v)))
                    }
                },
                _ => None,
            }?;

            Some((path, quote!( pub const #name: #type_and_value;)))
        })
        .collect()
}

pub fn pipeline_overridable_constants<F>(module: &naga::Module, demangle: F) -> TokenStream
where
    F: Fn(&str) -> TypePath + Clone,
{
    let overrides: Vec<_> = module.overrides.iter().map(|(_, o)| o).collect();

    let fields: Vec<_> = overrides
        .iter()
        .map(|o| {
            let name = Ident::new(o.name.as_ref().unwrap(), Span::call_site());
            // TODO: Do we only need to handle scalar types here?
            let ty = rust_type(
                &TypePath {
                    parent: ModulePath::default(),
                    name: String::new(),
                },
                module,
                &module.types[o.ty],
                MatrixVectorTypes::Rust,
                demangle.clone(),
            );

            if o.init.is_some() {
                quote!(pub #name: Option<#ty>)
            } else {
                quote!(pub #name: #ty)
            }
        })
        .collect();

    let required_entries: Vec<_> = overrides
        .iter()
        .filter_map(|o| {
            if o.init.is_some() {
                None
            } else {
                let key = override_key(o);

                let name = Ident::new(o.name.as_ref().unwrap(), Span::call_site());

                // TODO: Do we only need to handle scalar types here?
                let ty = &module.types[o.ty];
                let value = if matches!(ty.inner, naga::TypeInner::Scalar(s) if s.kind == naga::ScalarKind::Bool) {
                    quote!(if self.#name { 1.0 } else { 0.0})
                } else {
                    quote!(self.#name as f64)
                };

                Some(quote!((#key, #value)))
            }
        })
        .collect();

    // Add code for optionally inserting the constants with defaults.
    // Omitted constants will be initialized using the values defined in WGSL.
    let insert_optional_entries: Vec<_> = overrides
        .iter()
        .filter_map(|o| {
            if o.init.is_some() {
                let key = override_key(o);

                // TODO: Do we only need to handle scalar types here?
                let ty = &module.types[o.ty];
                let value = if matches!(ty.inner, naga::TypeInner::Scalar(s) if s.kind == naga::ScalarKind::Bool) {
                    quote!(if value { 1.0 } else { 0.0})
                } else {
                    quote!(value as f64)
                };

                let name = Ident::new(o.name.as_ref().unwrap(), Span::call_site());

                Some(quote! {
                    if let Some(value) = self.#name {
                        entries.push((#key, #value));
                    }
                })
            } else {
                None
            }
        })
        .collect();

    let init_entries = if insert_optional_entries.is_empty() {
        quote!(let entries = vec![#(#required_entries),*];)
    } else {
        quote!(let mut entries = vec![#(#required_entries),*];)
    };

    if !fields.is_empty() {
        // Create a Rust struct that can initialize the constants dictionary.
        quote! {
            pub struct OverrideConstants {
                #(#fields),*
            }

            impl OverrideConstants {
                pub fn constants(&self) -> Vec<(&'static str, f64)> {
                    #init_entries
                    #(#insert_optional_entries);*
                    entries
                }
            }
        }
    } else {
        quote!()
    }
}

fn override_key(o: &naga::Override) -> String {
    // The @id(id) should be the name if present.
    o.id.map(|i| i.to_string())
        .unwrap_or(o.name.clone().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{assert_tokens_eq, demangle_identity};
    use indoc::indoc;

    #[test]
    fn write_global_constants() {
        let source = indoc! {r#"
            enable f16;

            const INT_CONST = -12;
            
            const UNSIGNED_CONST = 34u;

            const FLOAT_CONST = 0.1;

            const SMALL_FLOAT_CONST: f16 = 0.25h;

            const BOOL_CONST = true;

            @fragment
            fn main() -> f32 {
                // TODO: This is valid WGSL syntax, but naga doesn't support it apparently.
                // const C_INNER = 456;

                if BOOL_CONST { 
                    return f32(INT_CONST) * f32(UNSIGNED_CONST) * FLOAT_CONST * f32(SMALL_FLOAT_CONST);
                } else {
                    return 0.0;
                }
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let consts = consts(&module, demangle_identity);
        let consts = consts.iter().map(|(_, c)| c);
        let actual = quote!(#(#consts)*);

        // TODO: Why are int and float consts missing?
        assert_tokens_eq!(
            quote! {
                pub const UNSIGNED_CONST: u32 = 34u32;
                pub const SMALL_FLOAT_CONST: half::f16 = half::f16::from_f32_const(0.25f32);
                pub const BOOL_CONST: bool = true;
            },
            actual
        );
    }

    #[test]
    fn write_pipeline_overrideable_constants_empty() {
        let source = indoc! {r#"
            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pipeline_overridable_constants(&module, demangle_identity);
        assert_tokens_eq!(quote!(), actual);
    }
}
