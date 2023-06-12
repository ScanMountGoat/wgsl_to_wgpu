use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub fn consts(module: &naga::Module) -> Vec<TokenStream> {
    // Create matching Rust structs for WGSL structs.
    // This is a UniqueArena, so each struct will only be generated once.
    module
        .constants
        .iter()
        .filter_map(|(_, t)| -> Option<TokenStream> {
            let name = Ident::new(t.name.as_ref()?, Span::call_site());
            let value = if let naga::ConstantInner::Scalar { value, .. } = &t.inner {
                *value
            } else {
                return None;
            };
            // There is a lot of downcasting from 64 to 32 here,
            // because the actual WGSL types are always i32, u32,
            // f32, f16 or bool. We don't really care about f16,
            // so that can be represented as f32.
            let type_and_value = match value {
                naga::ScalarValue::Sint(v) => {
                    let v = v as i32;
                    quote!(i32 = #v)
                }
                naga::ScalarValue::Uint(v) => {
                    let v = v as u32;
                    quote!(u32 = #v)
                }
                naga::ScalarValue::Float(v) => {
                    let v = v as f32;
                    quote!(f32 = #v)
                }
                naga::ScalarValue::Bool(v) => quote!(bool = #v),
            };

            Some(quote!( pub const #name: #type_and_value;))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assert_tokens_eq;
    use indoc::indoc;

    #[test]
    fn write_global_consts() {
        let source = indoc! {r#"
            const INT_CONST = 12;
            const UNSIGNED_CONST = 34u;
            const FLOAT_CONST = 0.1;
            // TODO: Naga doesn't implement f16, even though it's in the WGSL spec
            // const SMALL_FLOAT_CONST:f16 = 0.1h;
            const BOOL_CONST = true;

            @fragment
            fn main() {
                // TODO: This is valid WGSL syntax, but naga doesn't support it apparently.
                // const C_INNER = 456;
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let consts = consts(&module);
        let actual = quote!(#(#consts)*);
        eprintln!("{actual}");

        assert_tokens_eq!(
            quote! {
                pub const INT_CONST: i32 = 12i32;
                pub const UNSIGNED_CONST: u32 = 34u32;
                pub const FLOAT_CONST: f32 = 0.1f32;
                pub const BOOL_CONST: bool = true;
            },
            actual
        );
    }
}
