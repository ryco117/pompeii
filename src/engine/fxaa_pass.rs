use super::{shaders::ENTRY_POINT_MAIN, utils};

/// Shader for texture-mapping the entire screen. Useful for post-processing and fullscreen effects.
const FXAA_FRAGMENT: &[u32] =
    inline_spirv::include_spirv!("src/shaders/fxaa_frag.glsl", frag, glsl);

/// Define the push constants that are used in the fragment shader of the FXAA algorithm.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
    /// The width and height of the render surface, as `f32`s.
    pub screen_size: [f32; 2],
}

/// Define the pipeline used for the FXAA post-processing effect.
struct FxaaPipeline {
    render_pass: ash::vk::RenderPass,
    layout: ash::vk::PipelineLayout,
    pipeline: ash::vk::Pipeline,
    descriptor_set_layout: ash::vk::DescriptorSetLayout,
    fullscreen_vert_shader: ash::vk::ShaderModule,
    fxaa_frag_shader: ash::vk::ShaderModule,
}

impl FxaaPipeline {
    /// Create a new graphics pipeline for the FXAA post-processing effect.
    pub fn new(
        device: &ash::Device,
        render_pass: ash::vk::RenderPass,
        sampler: ash::vk::Sampler,
    ) -> Self {
        // Create the shader modules for the vertex and fragment shaders.
        // TODO: Allow caching of these shader modules.
        let fullscreen_vert_shader =
            utils::create_shader_module(device, super::shaders::FULLSCREEN_VERTEX);
        let fxaa_frag_shader = utils::create_shader_module(device, FXAA_FRAGMENT);
        let shader_stages = [
            ash::vk::PipelineShaderStageCreateInfo {
                stage: ash::vk::ShaderStageFlags::VERTEX,
                module: fullscreen_vert_shader,
                p_name: ENTRY_POINT_MAIN.as_ptr(),
                ..Default::default()
            },
            ash::vk::PipelineShaderStageCreateInfo {
                stage: ash::vk::ShaderStageFlags::FRAGMENT,
                module: fxaa_frag_shader,
                p_name: ENTRY_POINT_MAIN.as_ptr(),
                ..Default::default()
            },
        ];

        // Define the push constants that will be used by the fragment shader.
        let push_constants_range = ash::vk::PushConstantRange {
            stage_flags: ash::vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        // Define the pipeline viewport and scissor rectangle.
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        // Define the pipeline input assembly state.
        let input_state = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(ash::vk::PrimitiveTopology::TRIANGLE_STRIP);

        // Use the default rasterizer state to render the full screen mesh.
        let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo::default().line_width(1.);

        // Ensure that the pipeline does not use any multisampling.
        let sampling = ash::vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);

        // Define the color blend state.
        // TODO: This is likely overkill for FXAA post-processing.
        let color_blend_attachment = ash::vk::PipelineColorBlendAttachmentState {
            src_color_blend_factor: ash::vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: ash::vk::BlendOp::ADD, // This is the default.
            src_alpha_blend_factor: ash::vk::BlendFactor::SRC_ALPHA,
            dst_alpha_blend_factor: ash::vk::BlendFactor::DST_ALPHA,
            alpha_blend_op: ash::vk::BlendOp::ADD,
            color_write_mask: ash::vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };
        let color_blend_state = ash::vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        // Create the descriptor set layout.
        let descriptor_set_layout = {
            let descriptor_set_info = ash::vk::DescriptorSetLayoutCreateInfo {
                binding_count: 1,
                p_bindings: &ash::vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: ash::vk::ShaderStageFlags::FRAGMENT,
                    p_immutable_samplers: &sampler, // NOTE: This "array" must be equal in size to `descriptor_count`.
                    ..Default::default()
                },
                ..Default::default()
            };

            unsafe { device.create_descriptor_set_layout(&descriptor_set_info, None) }
                .expect("Failed to create descriptor set layout for FXAA post-processing")
        };

        // Create the pipeline layout.
        let pipeline_layout = {
            let layout_info = ash::vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constants_range,
                ..Default::default()
            };

            unsafe { device.create_pipeline_layout(&layout_info, None) }
                .expect("Failed to create pipeline layout for FXAA post-processing")
        };

        // Use dynamic states for the viewport and scissor rectangles.
        let dynamic_states = ash::vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: 2,
            p_dynamic_states: [
                ash::vk::DynamicState::VIEWPORT,
                ash::vk::DynamicState::SCISSOR,
            ]
            .as_ptr(),
            ..Default::default()
        };

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
    pipeline: FxaaPipeline,
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
    /// Create a new FXAA render pass and associated resources.
    pub fn new(
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
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
                address_mode_u: ash::vk::SamplerAddressMode::REPEAT,
                address_mode_v: ash::vk::SamplerAddressMode::REPEAT,
                address_mode_w: ash::vk::SamplerAddressMode::REPEAT,
                ..Default::default()
            };
            unsafe { device.create_sampler(&sampler_info, None) }
                .expect("Failed to create sampler for FXAA post-processing")
        };

        // Create the FXAA render pass and graphics pipeline.
        let render_pass = Self::create_render_pass(device, swapchain_format, destination_layout);
        let pipeline = FxaaPipeline::new(device, render_pass, sampler);

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
                descriptor_count: 1 as u32,
            }];
            let pool_info = ash::vk::DescriptorPoolCreateInfo {
                max_sets: framebuffers.len() as u32,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };
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
        allocator: &mut gpu_allocator::vulkan::Allocator,
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
        // Define the color attachment for the render pass.
        let attachment = ash::vk::AttachmentDescription {
            format: swapchain_format,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            load_op: ash::vk::AttachmentLoadOp::CLEAR,
            store_op: ash::vk::AttachmentStoreOp::STORE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            final_layout: destination_layout,
            ..Default::default()
        };
        let color_attachment_reference = ash::vk::AttachmentReference {
            attachment: 0,
            layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        // Define the single subpass that will be used in the render pass.
        let subpass_description = ash::vk::SubpassDescription {
            pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_reference,
            ..Default::default()
        };

        // Create the render pass.
        let render_pass_info = ash::vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &attachment,
            subpass_count: 1,
            p_subpasses: &subpass_description,
            dependency_count: 1,
            p_dependencies: &ash::vk::SubpassDependency {
                src_subpass: ash::vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: ash::vk::AccessFlags::empty(),
                dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ..Default::default()
            },
            ..Default::default()
        };
        unsafe { device.create_render_pass(&render_pass_info, None) }
            .expect("Failed to create render pass for FXAA post-processing")
    }

    /// Create new framebuffers that will be used during the FXAA render.
    fn create_framebuffers(
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
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
                    utils::create_image(device, memory_allocator, &image_info, "FXAA Image");
                let image_view = utils::create_image_view(device, image, swapchain_format, 1);

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
        let descriptor_set_info = ash::vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &descriptor_set_layout,
            ..Default::default()
        };
        internal_image_views
            .into_iter()
            .map(|image_view| {
                let set = unsafe { device.allocate_descriptor_sets(&descriptor_set_info) }
                    .expect("Failed to allocate descriptor set for FXAA post-processing")
                    .first()
                    .expect("FXAA descriptor set allocation returned an empty list")
                    .clone();

                // Update the descriptor set with the image attachment that will be sampled.
                unsafe {
                    device.update_descriptor_sets(
                        &[ash::vk::WriteDescriptorSet {
                            dst_set: set,
                            dst_binding: 0,
                            dst_array_element: 0,
                            descriptor_count: 1,
                            descriptor_type: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            p_image_info: &ash::vk::DescriptorImageInfo {
                                sampler,
                                image_view,
                                image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            },
                            ..Default::default()
                        }],
                        &[],
                    )
                };
                set
            })
            .collect()
    }

    /// Record the commands necessary to perform FXAA post-processing on the input image and write the result to the output image.
    pub fn render_frame(
        &self,
        device: &ash::Device,
        command_buffer: ash::vk::CommandBuffer,
        extent: ash::vk::Extent2D,
        image_index: usize,
        destination_layout: ash::vk::ImageLayout,
        destination_image: ash::vk::Image,
    ) {
        unsafe {
            // Use a pipeline barrier to ensure that we are able to read the input sampler in the correct layout.
            device.cmd_pipeline_barrier(
                command_buffer,
                ash::vk::PipelineStageFlags::TOP_OF_PIPE,
                ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &[ash::vk::ImageMemoryBarrier {
                    dst_access_mask: ash::vk::AccessFlags::SHADER_READ,
                    old_layout: ash::vk::ImageLayout::UNDEFINED,
                    new_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: self.framebuffers[image_index].2,
                    subresource_range: ash::vk::ImageSubresourceRange {
                        aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );

            // Begin the render pass for the current frame.
            device.cmd_begin_render_pass(
                command_buffer,
                &ash::vk::RenderPassBeginInfo {
                    render_pass: self.pipeline.render_pass,
                    framebuffer: self.framebuffers[image_index].0,
                    render_area: ash::vk::Rect2D {
                        offset: ash::vk::Offset2D::default(),
                        extent,
                    },
                    clear_value_count: 1,
                    p_clear_values: &ash::vk::ClearValue::default(),
                    ..Default::default()
                },
                ash::vk::SubpassContents::INLINE,
            );
        }

        // Set the shader push constants.
        let push_constants = PushConstants {
            screen_size: [extent.width as f32, extent.height as f32],
        };
        unsafe {
            device.cmd_push_constants(
                command_buffer,
                self.pipeline.layout,
                ash::vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    (&push_constants as *const PushConstants).cast(),
                    std::mem::size_of::<PushConstants>(),
                ),
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

        // Use a pipeline barrier to ensure that the temporary image is in the correct layout for the next render pass.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
                ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &[ash::vk::ImageMemoryBarrier {
                    src_access_mask: ash::vk::AccessFlags::SHADER_READ,
                    old_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    image: self.framebuffers[image_index].2,
                    subresource_range: ash::vk::ImageSubresourceRange {
                        aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }
    }

    /// Recreate the framebuffers used by this FXAA render pass instance, as well as the resources that use them.
    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
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
