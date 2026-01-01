use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::TypePath;

pub fn compute_module<F>(module: &naga::Module, demangle: F) -> TokenStream
where
    F: Fn(&str) -> TypePath + Clone,
{
    let entry_points: Vec<_> = module
        .entry_points
        .iter()
        .filter_map(|e| {
            if e.stage == naga::ShaderStage::Compute {
                let workgroup_size_constant = workgroup_size(e, demangle.clone());
                let create_pipeline = create_compute_pipeline(module, e, demangle.clone());

                Some(quote! {
                    #workgroup_size_constant
                    #create_pipeline
                })
            } else {
                None
            }
        })
        .collect();

    if entry_points.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod compute {
                #(#entry_points)*
            }
        }
    }
}

fn create_compute_pipeline<F>(
    module: &naga::Module,
    e: &naga::EntryPoint,
    demangle: F,
) -> TokenStream
where
    F: Fn(&str) -> TypePath,
{
    // Ignore the path, because the compute pipeline requires
    // the bind groups and layouts from the shader module.
    let name = &demangle(&e.name).name;

    // Compute pipeline creation has few parameters and can be generated.
    let pipeline_name = Ident::new(&format!("create_{name}_pipeline"), Span::call_site());

    // The entry name string itself should remain mangled to match the WGSL code.
    let entry_point = &e.name;

    // TODO: Include a user supplied module name in the label?
    let label = format!("Compute Pipeline {name}");

    if !module.overrides.is_empty() {
        quote! {
            pub fn #pipeline_name(device: &wgpu::Device, overrides: &super::OverrideConstants) -> wgpu::ComputePipeline {
                let module = super::create_shader_module(device);
                let layout = super::create_pipeline_layout(device);
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(#label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(#entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &overrides.constants(),
                        ..Default::default()
                    },
                    cache: Default::default(),
                })
            }
        }
    } else {
        quote! {
            pub fn #pipeline_name(device: &wgpu::Device) -> wgpu::ComputePipeline {
                let module = super::create_shader_module(device);
                let layout = super::create_pipeline_layout(device);
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(#label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(#entry_point),
                    compilation_options: Default::default(),
                    cache: Default::default(),
                })
            }
        }
    }
}

fn workgroup_size<F>(e: &naga::EntryPoint, demangle: F) -> TokenStream
where
    F: Fn(&str) -> TypePath + Clone,
{
    // Ignore the path, see above for reason.
    let name = &demangle(&e.name).name;

    let name = Ident::new(
        &format!("{}_WORKGROUP_SIZE", name.to_uppercase()),
        Span::call_site(),
    );
    let [x, y, z] = e
        .workgroup_size
        .map(|s| Literal::usize_unsuffixed(s as usize));
    quote!(pub const #name: [u32; 3] = [#x, #y, #z];)
}
