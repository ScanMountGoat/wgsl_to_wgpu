use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};

/// The enhanced render pass is a wrapper around RenderPass that gives us type level
/// state tailored to our particular shader. This lets us not only assure that our bind
/// groups and vertex buffers are set before we can draw, but also prevents us from
/// setting these in out-of-bounds slots or indexes.
pub fn enhanced_render_pass(qty_vertex_buffers: usize, qty_bind_groups: usize) -> TokenStream {

    let vertex_buffers = (0..qty_vertex_buffers)
        .map(|idx| {
            let vb = format_ident!("VB{}", idx);
            quote!(#vb)
        })
        .collect::<Vec<_>>();
    let bind_groups = (0..qty_bind_groups)
        .map(|idx| {
            let bg = format_ident!("BG{}", idx);
            quote!(#bg)
        }).collect::<Vec<_>>();
    let mut type_params = vertex_buffers.clone();
    type_params.append(&mut bind_groups.clone());


    let needing_vertex_buffers = (0..qty_vertex_buffers)
        .map(|idx| {
            let vb = format_ident!("NeedsVertexBuffer{}", idx);
            quote!(#vb)
        })
        .collect::<Vec<_>>();
    let needing_bind_groups = (0..qty_bind_groups)
        .map(|idx| {
            let bg = format_ident!("NeedsBindGroup{}", idx);
            quote!(#bg)
        }).collect::<Vec<_>>();
    let mut type_args_needy = needing_vertex_buffers.clone();
    type_args_needy.append(&mut needing_bind_groups.clone());
    let needy_structs = type_args_needy.clone().iter().map(|ts| quote!(pub struct #ts;)).collect::<Vec<_>>();


    let vertex_setters = vertex_buffers.clone().iter().enumerate().map(|(i, type_param)| {
        let fn_name = format_ident!("set_vertex_buffer_{}", i);
        let mut type_params_after = type_params.clone();
        type_params_after[i] = quote!(Ready);
        let slot = Literal::u32_unsuffixed(i as u32);
        quote! {
            pub fn #fn_name(mut self, buffer_slice: wgpu::BufferSlice<'rp>) -> EnhancedRenderPass<'rp, #(#type_params_after),*> {
                self.render_pass.set_vertex_buffer(#slot, buffer_slice);
                EnhancedRenderPass {
                    render_pass: self.render_pass,
                    type_state: std::marker::PhantomData,
                }
            }
        }
    }).collect::<Vec<TokenStream>>();

    let bind_group_setters = bind_groups.clone().iter().enumerate().map(|(i, type_param)| {
        let fn_name = format_ident!("set_bind_group_{}", i);
        let mut type_params_after = type_params.clone();
        type_params_after[vertex_buffers.len() + i] = quote!(Ready);
        let bg_name = format_ident!("BindGroup{}", i);
        quote! {
            pub fn #fn_name(mut self, bind_group: &'rp bind_groups::#bg_name) -> EnhancedRenderPass<'rp, #(#type_params_after),*> {
                bind_group.set(&mut self.render_pass);
                EnhancedRenderPass {
                    render_pass: self.render_pass,
                    type_state: std::marker::PhantomData,
                }
            }
        }
    }).collect::<Vec<TokenStream>>();

    let type_args_all_ready = (0..type_params.len()).map(|_| quote!(Ready)).collect::<Vec<_>>();

    quote! {
        #(#needy_structs)*
        pub struct Ready;
        pub struct EnhancedRenderPass<'rp, #(#type_params),*> {
            render_pass: wgpu::RenderPass<'rp>,
            type_state: std::marker::PhantomData<(#(#type_params),*)>,
        }
        impl<'rp> EnhancedRenderPass<'rp, #(#type_args_needy),*> {
            pub fn new(encoder: &'rp mut wgpu::CommandEncoder, desc: &wgpu::RenderPassDescriptor<'rp, '_>) -> EnhancedRenderPass<'rp, #(#type_args_needy),*> {
                let render_pass = encoder.begin_render_pass(desc);
                EnhancedRenderPass {
                    render_pass,
                    type_state: std::marker::PhantomData,
                }
            }
        }
        impl<'rp, #(#type_params),*> EnhancedRenderPass<'rp, #(#type_params),*> {

            // Tentative Goal:
            // Adequately wrap RenderPass that this method is almost never necessary.
            //
            // Restraint:
            // Never fully get rid of this method because there are probably lots
            // of extensions on RenderPass out there that people want to keep access to.
            pub fn inner(&mut self) -> &mut wgpu::RenderPass<'rp> {
                &mut self.render_pass
            }

            #(#vertex_setters)*

            #(#bind_group_setters)*
        }
        impl<'rp> EnhancedRenderPass<'rp, #(#type_args_all_ready),*> {
            pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
                self.render_pass.draw(vertices, instances);
            }
        }
    }
}