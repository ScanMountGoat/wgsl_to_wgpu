use case::CaseExt;
use naga::ShaderStage;
use naga::{Function, Module};
use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::wgsl::vertex_entry_structs;

pub fn fragment_target_count(module: &Module, f: &Function) -> usize {
    match &f.result {
        Some(r) => match &r.binding {
            Some(b) => {
                // Builtins don't have render targets.
                if matches!(b, naga::Binding::Location { .. }) {
                    1
                } else {
                    0
                }
            }
            None => {
                // Fragment functions should return a single variable or a struct.
                match &module.types[r.ty].inner {
                    naga::TypeInner::Struct { members, .. } => members
                        .iter()
                        .filter(|m| matches!(m.binding, Some(naga::Binding::Location { .. })))
                        .count(),
                    _ => 0,
                }
            }
        },
        None => 0,
    }
}

pub fn entry_point_constants(module: &naga::Module) -> TokenStream {
    let entry_points: Vec<TokenStream> = module
        .entry_points
        .iter()
        .map(|entry_point| {
            let entry_name = Literal::string(&entry_point.name);
            let const_name = Ident::new(
                &format!("ENTRY_{}", &entry_point.name.to_uppercase()),
                Span::call_site(),
            );
            quote! {
                pub const #const_name: &str = #entry_name;
            }
        })
        .collect();

    quote! {
        #(#entry_points)*
    }
}

pub fn vertex_states(module: &naga::Module) -> TokenStream {
    let vertex_entries: Vec<TokenStream> = module
        .entry_points
        .iter()
        .filter_map(|entry_point| match &entry_point.stage {
            ShaderStage::Vertex => {
                let fn_name =
                    Ident::new(&format!("{}_entry", &entry_point.name), Span::call_site());
                let const_name = Ident::new(
                    &format!("ENTRY_{}", &entry_point.name.to_uppercase()),
                    Span::call_site(),
                );

                let vertex_inputs = vertex_entry_structs(entry_point, module);
                let mut step_mode_params = vec![];
                let layout_expressions: Vec<TokenStream> = vertex_inputs
                    .iter()
                    .map(|input| {
                        let name = Ident::new(&input.name, Span::call_site());
                        let step_mode = Ident::new(&input.name.to_snake(), Span::call_site());
                        step_mode_params.push(quote!(#step_mode: wgpu::VertexStepMode));
                        quote!(#name::vertex_buffer_layout(#step_mode))
                    })
                    .collect();

                let n = Literal::usize_unsuffixed(vertex_inputs.len());

                let overrides = if !module.overrides.is_empty() {
                    Some(quote!(overrides: &OverrideConstants))
                } else {
                    None
                };

                let constants = if !module.overrides.is_empty() {
                    quote!(overrides.constants())
                } else {
                    quote!(Default::default())
                };

                let params = if step_mode_params.is_empty() {
                    quote!(#overrides)
                } else {
                    quote!(#(#step_mode_params),*, #overrides)
                };

                Some(quote! {
                    pub fn #fn_name(#params) -> VertexEntry<#n> {
                        VertexEntry {
                            entry_point: #const_name,
                            buffers: [
                                #(#layout_expressions),*
                            ],
                            constants: #constants
                        }
                    }
                })
            }
            _ => None,
        })
        .collect();

    // Don't generate unused code.
    if vertex_entries.is_empty() {
        quote!()
    } else {
        quote! {
            #[derive(Debug)]
            pub struct VertexEntry<const N: usize> {
                pub entry_point: &'static str,
                pub buffers: [wgpu::VertexBufferLayout<'static>; N],
                pub constants: std::collections::HashMap<String, f64>,
            }

            pub fn vertex_state<'a, const N: usize>(
                module: &'a wgpu::ShaderModule,
                entry: &'a VertexEntry<N>,
            ) -> wgpu::VertexState<'a> {
                wgpu::VertexState {
                    module,
                    entry_point: entry.entry_point,
                    buffers: &entry.buffers,
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &entry.constants,
                        ..Default::default()
                    },
                }
            }

            #(#vertex_entries)*
        }
    }
}

pub fn vertex_struct_methods(module: &naga::Module) -> TokenStream {
    let structs = vertex_input_structs(module);
    quote!(#(#structs)*)
}

fn vertex_input_structs(module: &naga::Module) -> Vec<TokenStream> {
    let vertex_inputs = crate::wgsl::get_vertex_input_structs(module);
    vertex_inputs.iter().map(|input|  {
        let name = Ident::new(&input.name, Span::call_site());

        let count = Literal::usize_unsuffixed(input.fields.len());
        let attributes: Vec<_> = input
            .fields
            .iter()
            .map(|(location, m)| {
                let field_name: TokenStream = m.name.as_ref().unwrap().parse().unwrap();
                let location = Literal::usize_unsuffixed(*location as usize);
                let format = crate::wgsl::vertex_format(&module.types[m.ty]);
                // TODO: Will the debug implementation always work with the macro?
                let format = Ident::new(&format!("{format:?}"), Span::call_site());

                quote! {
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::#format,
                        offset: std::mem::offset_of!(#name, #field_name) as u64,
                        shader_location: #location,
                    }
                }
            })
            .collect();


        // The vertex_attr_array! macro doesn't account for field alignment.
        // Structs with glam::Vec4 and glam::Vec3 fields will not be tightly packed.
        // Manually calculate the Rust field offsets to support using bytemuck for vertices.
        // This works since we explicitly mark all generated structs as repr(C).
        // Assume elements are in Rust arrays or slices, so use size_of for stride.
        // TODO: Should this enforce WebGPU alignment requirements for compatibility?
        // https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-gpuvertexbufferlayout

        // TODO: Support vertex inputs that aren't in a struct.
        quote! {
            impl #name {
                pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; #count] = [#(#attributes),*];

                pub const fn vertex_buffer_layout(step_mode: wgpu::VertexStepMode) -> wgpu::VertexBufferLayout<'static> {
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<#name>() as u64,
                        step_mode,
                        attributes: &#name::VERTEX_ATTRIBUTES
                    }
                }
            }
        }
    }).collect()
}

pub fn fragment_states(module: &naga::Module) -> TokenStream {
    let entries: Vec<TokenStream> = module
        .entry_points
        .iter()
        .filter_map(|entry_point| match &entry_point.stage {
            ShaderStage::Fragment => {
                let fn_name =
                    Ident::new(&format!("{}_entry", &entry_point.name), Span::call_site());

                let const_name = Ident::new(
                    &format!("ENTRY_{}", &entry_point.name.to_uppercase()),
                    Span::call_site(),
                );

                let target_count =
                    Literal::usize_unsuffixed(fragment_target_count(module, &entry_point.function));

                let overrides = if !module.overrides.is_empty() {
                    Some(quote!(overrides: &OverrideConstants))
                } else {
                    None
                };

                let constants = if !module.overrides.is_empty() {
                    quote!(overrides.constants())
                } else {
                    quote!(Default::default())
                };

                Some(quote! {
                    pub fn #fn_name(
                        targets: [Option<wgpu::ColorTargetState>; #target_count],
                        #overrides
                    ) -> FragmentEntry<#target_count> {
                        FragmentEntry {
                            entry_point: #const_name,
                            targets,
                            constants: #constants
                        }
                    }
                })
            }
            _ => None,
        })
        .collect();

    // Don't generate unused code.
    if entries.is_empty() {
        quote!()
    } else {
        quote! {
            #[derive(Debug)]
            pub struct FragmentEntry<const N: usize> {
                pub entry_point: &'static str,
                pub targets: [Option<wgpu::ColorTargetState>; N],
                pub constants: std::collections::HashMap<String, f64>,
            }

            pub fn fragment_state<'a, const N: usize>(
                module: &'a wgpu::ShaderModule,
                entry: &'a FragmentEntry<N>,
            ) -> wgpu::FragmentState<'a> {
                wgpu::FragmentState {
                    module,
                    entry_point: entry.entry_point,
                    targets: &entry.targets,
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &entry.constants,
                        ..Default::default()
                    },
                }
            }

            #(#entries)*
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::assert_tokens_eq;
    use indoc::indoc;

    #[test]
    fn write_vertex_states_no_entries() {
        let source = indoc! {r#"
            struct Input {
                @location(0) position: vec4<f32>,
            };
            @fragment
            fn main(in: Input) {}
        "#
        };

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_states(&module);

        assert_tokens_eq!(quote!(), actual)
    }

    #[test]
    fn write_fragment_states_multiple_entries() {
        let source = indoc! {r#"
            struct Output {
                @location(0) col0: vec4<f32>,
                @builtin(frag_depth) depth: f32,
                @location(1) col1: vec4<f32>,
            };
            @fragment
            fn fs_multiple() -> Output {}

            @fragment
            fn fs_single() -> @location(0) vec4<f32> {}

            @fragment
            fn fs_single_builtin() -> @builtin(frag_depth) f32 {}

            @fragment
            fn fs_empty() {}
        "#
        };

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = fragment_states(&module);

        assert_tokens_eq!(
            quote! {
                #[derive(Debug)]
                pub struct FragmentEntry<const N: usize> {
                    pub entry_point: &'static str,
                    pub targets: [Option<wgpu::ColorTargetState>; N],
                    pub constants: std::collections::HashMap<String, f64>,
                }
                pub fn fragment_state<'a, const N: usize>(
                    module: &'a wgpu::ShaderModule,
                    entry: &'a FragmentEntry<N>,
                ) -> wgpu::FragmentState<'a> {
                    wgpu::FragmentState {
                        module,
                        entry_point: entry.entry_point,
                        targets: &entry.targets,
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &entry.constants,
                            ..Default::default()
                        },
                    }
                }
                pub fn fs_multiple_entry(
                    targets: [Option<wgpu::ColorTargetState>; 2]
                ) -> FragmentEntry<2> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_MULTIPLE,
                        targets,
                        constants: Default::default(),
                    }
                }
                pub fn fs_single_entry(
                    targets: [Option<wgpu::ColorTargetState>; 1]
                ) -> FragmentEntry<1> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_SINGLE,
                        targets,
                        constants: Default::default(),
                    }
                }
                pub fn fs_single_builtin_entry(
                    targets: [Option<wgpu::ColorTargetState>; 0]
                ) -> FragmentEntry<0> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_SINGLE_BUILTIN,
                        targets,
                        constants: Default::default(),
                    }
                }
                pub fn fs_empty_entry(
                    targets: [Option<wgpu::ColorTargetState>; 0]
                ) -> FragmentEntry<0> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_EMPTY,
                        targets,
                        constants: Default::default(),
                    }
                }
            },
            actual
        )
    }

    #[test]
    fn write_fragment_states_single_entry() {
        let source = indoc! {r#"
            override test: bool = true;

            @fragment
            fn fs_single() -> @location(0) vec4<f32> {}
        "#
        };

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = fragment_states(&module);

        assert_tokens_eq!(
            quote! {
                #[derive(Debug)]
                pub struct FragmentEntry<const N: usize> {
                    pub entry_point: &'static str,
                    pub targets: [Option<wgpu::ColorTargetState>; N],
                    pub constants: std::collections::HashMap<String, f64>,
                }
                pub fn fragment_state<'a, const N: usize>(
                    module: &'a wgpu::ShaderModule,
                    entry: &'a FragmentEntry<N>,
                ) -> wgpu::FragmentState<'a> {
                    wgpu::FragmentState {
                        module,
                        entry_point: entry.entry_point,
                        targets: &entry.targets,
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &entry.constants,
                            ..Default::default()
                        },
                    }
                }
                pub fn fs_single_entry(
                    targets: [Option<wgpu::ColorTargetState>; 1],
                    overrides: &OverrideConstants
                ) -> FragmentEntry<1> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_SINGLE,
                        targets,
                        constants: overrides.constants(),
                    }
                }
            },
            actual
        )
    }
}
