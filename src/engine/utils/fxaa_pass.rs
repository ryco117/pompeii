use std::sync::Mutex;

use super::shaders::ENTRY_POINT_MAIN;

/// Shader for texture-mapping the entire screen. Useful for post-processing and fullscreen effects.
const FXAA_FRAGMENT: &[u32] =
    inline_spirv::include_spirv!("src/shaders/fxaa_frag.glsl", frag, glsl);

/// Define the push constants that are used in the fragment shader of the FXAA algorithm.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
pub struct PushConstants {
    /// The width and height of the render surface, as `f32`s.
    pub inverse_screen_size: [f32; 2],
}

/// Define the pipeline used for the FXAA post-processing effect.
struct Pipeline {
    render_pass: ash::vk::RenderPass,
    layout: ash::vk::PipelineLayout,
    pipeline: ash::vk::Pipeline,
    descriptor_set_layout: ash::vk::DescriptorSetLayout,
    fullscreen_vert_shader: ash::vk::ShaderModule,
    fxaa_frag_shader: ash::vk::ShaderModule,
}

impl Pipeline {
    /// Create a new graphics pipeline for the FXAA post-processing effect.
    pub fn new(
        device: &ash::Device,
        render_pass: ash::vk::RenderPass,
        sampler: ash::vk::Sampler,
    ) -> Self {
        // Create the shader modules for the vertex and fragment shaders.
        // TODO: Allow caching of these shader modules.
        let fullscreen_vert_shader =
            super::create_shader_module(device, super::shaders::FULLSCREEN_VERTEX);
        let fxaa_frag_shader = super::create_shader_module(device, FXAA_FRAGMENT);
        let shader_stages = [
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::VERTEX)
                .module(fullscreen_vert_shader)
                .name(ENTRY_POINT_MAIN),
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::FRAGMENT)
                .module(fxaa_frag_shader)
                .name(ENTRY_POINT_MAIN),
        ];

        // Define the push constants that will be used by the fragment shader.
        let push_constants_range = ash::vk::PushConstantRange {
            stage_flags: ash::vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        // Define the pipeline viewport and scissor rectangle.
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Define the pipeline input assembly state.
        let input_state = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(ash::vk::PrimitiveTopology::TRIANGLE_STRIP);

        // Use the default rasterizer state to render the full screen mesh.
        let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo::default().line_width(1.);

        // Ensure that the pipeline does not use any multisampling.
        let sampling = ash::vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);

        // Define the color blend state.
        let color_blend_attachment = [ash::vk::PipelineColorBlendAttachmentState {
            src_color_blend_factor: ash::vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: ash::vk::BlendOp::ADD, // This is the default.
            src_alpha_blend_factor: ash::vk::BlendFactor::SRC_ALPHA,
            dst_alpha_blend_factor: ash::vk::BlendFactor::DST_ALPHA,
            alpha_blend_op: ash::vk::BlendOp::ADD,
            color_write_mask: ash::vk::ColorComponentFlags::RGBA,
            ..Default::default()
        }];
        let color_blend_state = ash::vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment);

        // Create the descriptor set layout.
        let descriptor_set_layout = {
            let sampler = [sampler];
            let descriptor_set_info = ash::vk::DescriptorSetLayoutCreateInfo {
                binding_count: 1,
                p_bindings: &ash::vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(ash::vk::ShaderStageFlags::FRAGMENT)
                    .immutable_samplers(&sampler), // NOTE: This helper also sets the `descriptor_count` property to ensure valid usage.
                ..Default::default()
            };

            unsafe { device.create_descriptor_set_layout(&descriptor_set_info, None) }
                .expect("Failed to create descriptor set layout for FXAA post-processing")
        };

        // Create the pipeline layout.
        let pipeline_layout = {
            let descriptor_set_layout = [descriptor_set_layout];
            let push_constants_range = [push_constants_range];
            let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&descriptor_set_layout)
                .push_constant_ranges(&push_constants_range);

            unsafe { device.create_pipeline_layout(&layout_info, None) }
                .expect("Failed to create pipeline layout for FXAA post-processing")
        };

        // Use dynamic states for the viewport and scissor rectangles.
        let dynamic_states = ash::vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
            ash::vk::DynamicState::VIEWPORT,
            ash::vk::DynamicState::SCISSOR,
        ]);

        // Create the FXAA pipeline.
        let pipeline = {
            let pipeline_info = ash::vk::GraphicsPipelineCreateInfo {
                stage_count: shader_stages.len() as u32,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &ash::vk::PipelineVertexInputStateCreateInfo::default(),
                p_input_assembly_state: &input_state,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterizer,
                p_multisample_state: &sampling,
                p_color_blend_state: &color_blend_state,
                p_dynamic_state: &dynamic_states,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                base_pipeline_index: -1,
                ..Default::default()
            };

            unsafe {
                device.create_graphics_pipelines(
                    ash::vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
            }
            .expect("Failed to create graphics pipeline for FXAA post-processing")[0]
        };

        Self {
            render_pass,
            layout: pipeline_layout,
            pipeline,
            descriptor_set_layout,
            fullscreen_vert_shader,
            fxaa_frag_shader,
        }
    }
}

/// An implementation of a FXAA render pass.
/// This can be used as a post-processing effect in a larger render pipeline.
pub struct FxaaPass {
    pipeline: Pipeline,
    pub sampler: ash::vk::Sampler,
    pub framebuffers: Vec<(
        ash::vk::Framebuffer,
        ash::vk::ImageView,
        ash::vk::Image,
        gpu_allocator::vulkan::Allocation,
    )>,
    descriptor_pool: ash::vk::DescriptorPool,
    descriptor_sets: Vec<ash::vk::DescriptorSet>,
}

impl FxaaPass {
    const OPTIMAL_INTERNAL_IMAGE_LAYOUT: ash::vk::ImageLayout =
        ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

    /// Create a new FXAA render pass and associated resources.
    pub fn new(
        device: &ash::Device,
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        extent: ash::vk::Extent2D,
        swapchain_format: ash::vk::Format,
        destination_views: &[ash::vk::ImageView],
        destination_layout: ash::vk::ImageLayout,
    ) -> Self {
        // Create the sampler which will allow the FXAA shader to sample the input image.
        let sampler = {
            // NOTE: All of these values are the defaults, but we're setting them explicitly for clarity.
            let sampler_info = ash::vk::SamplerCreateInfo {
                mag_filter: ash::vk::Filter::NEAREST,
                min_filter: ash::vk::Filter::NEAREST,
                mipmap_mode: ash::vk::SamplerMipmapMode::NEAREST,
                address_mode_u: ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
                max_lod: ash::vk::LOD_CLAMP_NONE,
                ..Default::default()
            };
            unsafe { device.create_sampler(&sampler_info, None) }
                .expect("Failed to create sampler for FXAA post-processing")
        };

        // Create the FXAA render pass and graphics pipeline.
        let render_pass = Self::create_render_pass(device, swapchain_format, destination_layout);
        let pipeline = Pipeline::new(device, render_pass, sampler);

        // Create the framebuffers that will be used during this render.
        let framebuffers = Self::create_framebuffers(
            device,
            memory_allocator,
            extent,
            swapchain_format,
            destination_views,
            pipeline.render_pass,
        );

        // Create a descriptor pool and descriptor sets.
        let descriptor_pool = {
            let pool_sizes = [ash::vk::DescriptorPoolSize {
                ty: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: destination_views.len() as u32, // One descriptor per framebuffer.
            }];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(framebuffers.len() as u32)
                .pool_sizes(&pool_sizes);
            unsafe { device.create_descriptor_pool(&pool_info, None) }
                .expect("Failed to create descriptor pool for FXAA post-processing")
        };

        let descriptor_sets = Self::create_descriptor_sets(
            device,
            descriptor_pool,
            pipeline.descriptor_set_layout,
            framebuffers.iter().map(|(_, view, _, _)| *view),
            sampler,
        );

        Self {
            pipeline,
            sampler,
            framebuffers,
            descriptor_pool,
            descriptor_sets,
        }
    }

    /// Clean up the resources used by this FXAA render pass instance.
    pub fn destroy(
        &mut self,
        device: &ash::Device,
        allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
    ) {
        // Destroy the descriptor pool and descriptor sets.
        unsafe {
            device.reset_descriptor_pool(
                self.descriptor_pool,
                ash::vk::DescriptorPoolResetFlags::empty(),
            )
        }
        .expect("Failed to free FXAA descriptor sets");
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };

        // Destroy the framebuffers.
        for (framebuffer, image_view, image, allocation) in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
            unsafe { device.destroy_image_view(image_view, None) };
            unsafe { device.destroy_image(image, None) };
            allocator
                .lock()
                .expect("Failed to lock allocator in `FxaaPass::destroy`")
                .free(allocation)
                .expect("Failed to free FXAA image allocation");
        }

        // Destroy the pipeline.
        unsafe { device.destroy_pipeline(self.pipeline.pipeline, None) };
        unsafe { device.destroy_pipeline_layout(self.pipeline.layout, None) };
        unsafe { device.destroy_descriptor_set_layout(self.pipeline.descriptor_set_layout, None) };
        unsafe { device.destroy_render_pass(self.pipeline.render_pass, None) };

        // Destroy the shader modules.
        unsafe { device.destroy_shader_module(self.pipeline.fullscreen_vert_shader, None) };
        unsafe { device.destroy_shader_module(self.pipeline.fxaa_frag_shader, None) };

        // Destroy the sampler.
        unsafe { device.destroy_sampler(self.sampler, None) };
    }

    /// Helper to create a new render pass for the FXAA post-processing effect.
    fn create_render_pass(
        device: &ash::Device,
        swapchain_format: ash::vk::Format,
        destination_layout: ash::vk::ImageLayout,
    ) -> ash::vk::RenderPass {
        // Define the color attachment for the render pass, i.e., the output image of this pass.
        let attachment = [ash::vk::AttachmentDescription {
            format: swapchain_format,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            load_op: ash::vk::AttachmentLoadOp::DONT_CARE, // Each bit of the surface will be re-drawn so a clear is not necessary.
            store_op: ash::vk::AttachmentStoreOp::STORE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            final_layout: destination_layout,
            ..Default::default()
        }];
        let color_attachment_reference = [ash::vk::AttachmentReference {
            attachment: 0,
            layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        // Define the single subpass that will be used in the render pass.
        let subpass_description = [ash::vk::SubpassDescription::default()
            .pipeline_bind_point(ash::vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_reference)];

        // Define the subpass dependencies.
        let subpass_dependencies = [ash::vk::SubpassDependency {
            src_subpass: ash::vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: ash::vk::AccessFlags::NONE,
            dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        // Create the render pass.
        let render_pass_info = ash::vk::RenderPassCreateInfo::default()
            .attachments(&attachment)
            .subpasses(&subpass_description)
            .dependencies(&subpass_dependencies);
        unsafe { device.create_render_pass(&render_pass_info, None) }
            .expect("Failed to create render pass for FXAA post-processing")
    }

    /// Create new framebuffers that will be used during the FXAA render.
    fn create_framebuffers(
        device: &ash::Device,
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        extent: ash::vk::Extent2D,
        swapchain_format: ash::vk::Format,
        destination_views: &[ash::vk::ImageView],
        render_pass: ash::vk::RenderPass,
    ) -> Vec<(
        ash::vk::Framebuffer,
        ash::vk::ImageView,
        ash::vk::Image,
        gpu_allocator::vulkan::Allocation,
    )> {
        destination_views
            .iter()
            .map(|destination_view| {
                // Create an intermediate image to render to before FXAA processing. The final image will be the swapchain image.
                let image_info = ash::vk::ImageCreateInfo {
                    image_type: ash::vk::ImageType::TYPE_2D,
                    format: swapchain_format,
                    extent: ash::vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: ash::vk::SampleCountFlags::TYPE_1,
                    usage: ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | ash::vk::ImageUsageFlags::SAMPLED,
                    ..Default::default()
                };
                let (image, allocation) =
                    super::create_image(device, memory_allocator, &image_info, "FXAA Image");
                let image_view = super::create_image_view(device, image, swapchain_format, 1);

                let framebuffer_info = ash::vk::FramebufferCreateInfo {
                    render_pass,
                    attachment_count: 1,
                    p_attachments: destination_view,
                    width: extent.width,
                    height: extent.height,
                    layers: 1,
                    ..Default::default()
                };
                let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }
                    .expect("Failed to create framebuffer for FXAA post-processing");
                (framebuffer, image_view, image, allocation)
            })
            .collect()
    }

    /// Helper to create descriptor sets for post-processing a rendered image.
    fn create_descriptor_sets<I>(
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        descriptor_set_layout: ash::vk::DescriptorSetLayout,
        internal_image_views: I,
        sampler: ash::vk::Sampler,
    ) -> Vec<ash::vk::DescriptorSet>
    where
        I: IntoIterator<Item = ash::vk::ImageView>,
    {
        // Define the common layout for each descriptor set to create.
        let descriptor_set_layout = [descriptor_set_layout];
        let descriptor_set_info = ash::vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layout);

        // Create a descriptor set for each image view that will be sampled from during render passes.
        internal_image_views
            .into_iter()
            .map(|image_view| {
                let &set = unsafe { device.allocate_descriptor_sets(&descriptor_set_info) }
                    .expect("Failed to allocate descriptor set for FXAA post-processing")
                    .first()
                    .expect("FXAA descriptor set allocation returned an empty list");

                // Update the descriptor set with the image attachment that will be sampled.
                unsafe {
                    device.update_descriptor_sets(
                        &[ash::vk::WriteDescriptorSet {
                            dst_set: set,
                            dst_binding: 0,
                            dst_array_element: 0,
                            descriptor_count: 1, // The number of elements pointed to by `p_image_info`.
                            descriptor_type: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            p_image_info: &ash::vk::DescriptorImageInfo {
                                sampler,
                                image_view,
                                image_layout: Self::OPTIMAL_INTERNAL_IMAGE_LAYOUT,
                            },
                            ..Default::default()
                        }],
                        &[],
                    );
                };
                set
            })
            .collect()
    }

    /// Record the commands necessary to perform FXAA post-processing on the input image and write the result to the output image.
    /// # Safety
    /// * The command buffer must be in the recording state.
    /// * The image index must be a valid index into the framebuffers array.
    /// * The image index must specify an internal image that currently has layout `COLOR_ATTACHMENT_OPTIMAL`.
    pub fn render_frame(
        &self,
        device: &ash::Device,
        command_buffer: ash::vk::CommandBuffer,
        extent: ash::vk::Extent2D,
        image_index: usize,
    ) {
        unsafe {
            // Use a pipeline barrier to ensure that we are able to read the input sampler in the correct layout.
            let image_barrier = ash::vk::ImageMemoryBarrier2::default()
                .src_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(ash::vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(ash::vk::AccessFlags2::SHADER_READ)
                .old_layout(ash::vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .new_layout(ash::vk::ImageLayout::READ_ONLY_OPTIMAL)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(self.framebuffers[image_index].2)
                .subresource_range(
                    ash::vk::ImageSubresourceRange::default()
                        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier2(
                command_buffer,
                &ash::vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            );

            // Begin the render pass for the current frame.
            device.cmd_begin_render_pass(
                command_buffer,
                &ash::vk::RenderPassBeginInfo::default()
                    .render_pass(self.pipeline.render_pass)
                    .framebuffer(self.framebuffers[image_index].0)
                    .render_area(ash::vk::Rect2D {
                        offset: ash::vk::Offset2D::default(),
                        extent,
                    }),
                ash::vk::SubpassContents::INLINE,
            );
        }

        // Set the shader push constants.
        let push_constants = PushConstants {
            inverse_screen_size: [1. / extent.width as f32, 1. / extent.height as f32],
        };
        unsafe {
            device.cmd_push_constants(
                command_buffer,
                self.pipeline.layout,
                ash::vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
        }

        // Set the viewport and scissor in the command buffer because we specified they would be set dynamically in the pipeline.
        unsafe {
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[ash::vk::Viewport {
                    x: 0.,
                    y: 0.,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.,
                    max_depth: 1.,
                }],
            );
            device.cmd_set_scissor(
                command_buffer,
                0,
                &[ash::vk::Rect2D {
                    offset: ash::vk::Offset2D::default(),
                    extent,
                }],
            );
        }

        // Bind the graphics pipeline to the command buffer.
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
        }

        // Bind the descriptor set to the proper image attachment.
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
        }

        // Draw the full screen quad.
        unsafe { device.cmd_draw(command_buffer, 4, 1, 0, 0) };

        // End the render pass.
        unsafe { device.cmd_end_render_pass(command_buffer) };
    }

    /// Recreate the framebuffers used by this FXAA render pass instance, as well as the resources that use them.
    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        extent: ash::vk::Extent2D,
        image_format: ash::vk::Format,
        destination_views: &[ash::vk::ImageView],
    ) {
        // Destroy the descriptor pool and descriptor sets.
        unsafe {
            device.reset_descriptor_pool(
                self.descriptor_pool,
                ash::vk::DescriptorPoolResetFlags::empty(),
            )
        }
        .expect("Failed to free FXAA descriptor sets");

        // Destroy the framebuffers.
        for (framebuffer, image_view, image, allocation) in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
            unsafe { device.destroy_image_view(image_view, None) };
            unsafe { device.destroy_image(image, None) };
            memory_allocator
                .lock()
                .expect("Failed to lock allocator in `FxaaPass::recreate_framebuffers`")
                .free(allocation)
                .expect("Failed to free FXAA image allocation");
        }

        // Recreate the framebuffers.
        self.framebuffers = Self::create_framebuffers(
            device,
            memory_allocator,
            extent,
            image_format,
            destination_views,
            self.pipeline.render_pass,
        );

        // Recreate the descriptor sets.
        self.descriptor_sets = Self::create_descriptor_sets(
            device,
            self.descriptor_pool,
            self.pipeline.descriptor_set_layout,
            self.framebuffers.iter().map(|(_, view, _, _)| *view),
            self.sampler,
        );
    }

    /// Get the framebuffers and related data.
    pub fn framebuffer_data(
        &self,
    ) -> &[(
        ash::vk::Framebuffer,
        ash::vk::ImageView,
        ash::vk::Image,
        gpu_allocator::vulkan::Allocation,
    )] {
        &self.framebuffers
    }
}
