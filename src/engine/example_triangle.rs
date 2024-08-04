use crate::engine::utils::{self, shaders::ENTRY_POINT_MAIN};

/// Store the SPIR-V representation of the shaders in the binary.
/// This basic triangle example uses a vertex shader which stores it own vertices for simplicity.
/// Only the fragment shader takes in data from the application and it does so through a specialization constant and a push constant.
pub mod shaders {
    /// Standard triangle-example vertex shader.
    pub const BB_TRIANGLE_VERTEX: &[u32] =
        inline_spirv::include_spirv!("src/shaders/bb_triangle_vert.glsl", vert, glsl);

    /// Standard triangle-example fragment shader.
    pub const BB_TRIANGLE_FRAGMENT: &[u32] =
        inline_spirv::include_spirv!("src/shaders/bb_triangle_frag.glsl", frag, glsl);
}

/// Define the specialization constants that can be used with the shaders of this application.
/// Specifically, this shader only accepts a single `boolean32` that toggles the reflection of the triangle along the Y axis.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
pub struct SpecializationConstants {
    /// Reflect the vertices of the triangle along their Y axis.
    pub toggle: u32,
}

/// Define the push constants that can be used with the shaders of this application (i.e., `BB_TRIANGLE_FRAGMENT`).
/// Specifically, this shader only accepts a single float representing a time in seconds since the application start.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
pub struct PushConstants {
    /// The time in seconds since the application started.
    pub time: f32,
}

/// Create the render pass capable of orchestrating the rendering of framebuffers for this application.
fn create_render_pass(
    device: &ash::Device,
    image_format: ash::vk::Format,
    multisample: Option<ash::vk::SampleCountFlags>,
    destination_layout: ash::vk::ImageLayout,
) -> ash::vk::RenderPass {
    // Ensure that the multisample count is valid and not single-sampled.
    let multisample = multisample.filter(|&s| s != ash::vk::SampleCountFlags::TYPE_1);

    // Create the attachment descriptor defining the color attachment output from the fragment shader.
    let mut attachment_descriptors = vec![ash::vk::AttachmentDescription {
        format: image_format,
        samples: multisample.unwrap_or(ash::vk::SampleCountFlags::TYPE_1), // Default to single-sampled. Depth/stencil attachments must have the same sample count as the color attachments.
        load_op: ash::vk::AttachmentLoadOp::CLEAR, // Clear the framebuffer before rendering.
        store_op: if multisample.is_some() {
            // The multisample resolve attachment will store the final image, this image is transient.
            ash::vk::AttachmentStoreOp::DONT_CARE
        } else {
            ash::vk::AttachmentStoreOp::STORE // Store the framebuffer after rendering.
        },
        stencil_load_op: ash::vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: ash::vk::ImageLayout::UNDEFINED,
        final_layout: if multisample.is_some() {
            ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            destination_layout
        },
        ..Default::default()
    }];

    // Define the index the subpass will use to define their color attachment.
    let color_attachment_reference = ash::vk::AttachmentReference {
        attachment: 0, // NOTE: Uses the index of color attachment in framebuffer.
        layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    // When multisampling, create a resolve attachment to resolve the multisampled image to.
    // Note that there must be a resolve attachment for every color attachment.
    let mut resolve_attachment_reference = None;
    if multisample.is_some() {
        let index = attachment_descriptors.len();
        attachment_descriptors.push(ash::vk::AttachmentDescription {
            format: image_format,
            samples: ash::vk::SampleCountFlags::TYPE_1, // The resolve attachment must be single-sampled.
            load_op: ash::vk::AttachmentLoadOp::DONT_CARE,
            store_op: ash::vk::AttachmentStoreOp::STORE,
            stencil_load_op: ash::vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            final_layout: destination_layout,
            ..Default::default()
        });

        resolve_attachment_reference = Some(ash::vk::AttachmentReference {
            attachment: index as u32,
            layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        });
    }

    // Define the single subpass that will be used in the render pass.
    let subpass_description = ash::vk::SubpassDescription {
        pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &color_attachment_reference,
        p_resolve_attachments: resolve_attachment_reference
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref),
        ..Default::default()
    };

    let subpass_dependencies = [ash::vk::SubpassDependency {
        src_subpass: ash::vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: if multisample.is_some() {
            // The resolve attachment will be written to at the end of a previous subpass.
            // Ensure that we wait for the attachment to be ready.
            ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        } else {
            ash::vk::AccessFlags::NONE
        },
        dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ..Default::default()
    }];

    unsafe {
        device.create_render_pass(
            &ash::vk::RenderPassCreateInfo {
                attachment_count: attachment_descriptors.len() as u32,
                p_attachments: attachment_descriptors.as_ptr(),
                subpass_count: 1,
                p_subpasses: &subpass_description,
                dependency_count: subpass_dependencies.len() as u32,
                p_dependencies: subpass_dependencies.as_ptr(),
                ..Default::default()
            },
            None,
        )
    }
    .expect("Unable to create the application render pass")
}

/// Create a framebuffer that can be used with this render pass.
pub fn create_framebuffer(
    device: &ash::Device,
    render_pass: ash::vk::RenderPass,
    image_views: &[ash::vk::ImageView],
    extent: ash::vk::Extent2D,
) -> ash::vk::Framebuffer {
    let ash::vk::Extent2D { width, height } = extent;
    unsafe {
        device.create_framebuffer(
            &ash::vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: image_views.len() as u32,
                p_attachments: image_views.as_ptr(),
                width,
                height,
                layers: 1,
                ..Default::default()
            },
            None,
        )
    }
    .expect("Unable to create framebuffer")
}

/// Create all framebuffers needed for this example, including optionally resolving a multisampled color attachment and applying a post-processing effect (FXAA).
pub fn create_framebuffers(
    device: &ash::Device,
    swapchain: &utils::Swapchain,
    render_pass: ash::vk::RenderPass,
    fxaa_pass: Option<&utils::fxaa_pass::FxaaPass>,
) -> Vec<ash::vk::Framebuffer> {
    let extent = swapchain.extent();

    // Render to a temporary image if FXAA is enabled, otherwise render to the swapchain's images.
    let destination_image_views = if let Some(fxaa_pass) = &fxaa_pass {
        fxaa_pass
            .framebuffer_data()
            .iter()
            .map(|(_, img, _, _)| *img)
            .collect()
    } else {
        swapchain.image_views().to_vec()
    };

    // Render to a multisampled image if MSAA is enabled, otherwise render directly to the destination image.
    if let Some(multisample_views) = swapchain.multisample_views() {
        // Note that the order of the image views must be the same as specified during the render pass creation.
        destination_image_views
            .iter()
            .zip(multisample_views.iter())
            .map(|(destination_view, multisample_view)| {
                create_framebuffer(
                    device,
                    render_pass,
                    &[*multisample_view, *destination_view],
                    extent,
                )
            })
            .collect()
    } else {
        destination_image_views
            .iter()
            .map(|destination_view| {
                create_framebuffer(device, render_pass, &[*destination_view], extent)
            })
            .collect()
    }
}

/// Store the shader modules used by this simple example.
pub struct Shaders {
    vertex_module: ash::vk::ShaderModule,
    fragment_module: ash::vk::ShaderModule,
}

/// Manage the graphics pipeline and dependent resources.
pub struct Pipeline {
    handle: ash::vk::Pipeline,
    layout: ash::vk::PipelineLayout,
    specialization_constants: SpecializationConstants,
    render_pass: ash::vk::RenderPass,
    shaders: Shaders,
    framebuffers: Vec<ash::vk::Framebuffer>,
}

/// Allow the caller to create a new pipeline from either an existing render pass or the information to create a new one.
#[derive(Clone, Copy, Debug)]
pub enum CreateReuseRenderPass {
    Create {
        image_format: ash::vk::Format,
        destination_layout: ash::vk::ImageLayout,
    },
    Reuse(ash::vk::RenderPass),
}

impl Pipeline {
    /// Create the graphics pipeline capable of rendering this application's scene.
    //  TODO: Create a type to help configure the pipeline creation.
    pub fn new(
        device: &ash::Device,
        vertex_module: Option<ash::vk::ShaderModule>,
        fragment_module: Option<ash::vk::ShaderModule>,
        create_or_reuse_render_pass: CreateReuseRenderPass,
        swapchain: &utils::Swapchain,
        specialization_constants: SpecializationConstants,
        fxaa_pass: Option<&utils::fxaa_pass::FxaaPass>,
    ) -> Self {
        // Determine whether the swapchain contains additional multisampled images.
        let multisample_count = swapchain.multisample_count();

        // Create or reuse the render pass for this pipeline.
        let render_pass = match create_or_reuse_render_pass {
            CreateReuseRenderPass::Create {
                image_format,
                destination_layout,
            } => create_render_pass(device, image_format, multisample_count, destination_layout),
            CreateReuseRenderPass::Reuse(render_pass) => render_pass,
        };

        // Define the specialization constants used for this pipeline creation.
        let specialization_map_toggle = [ash::vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: std::mem::size_of::<SpecializationConstants>(),
        }];
        let vertex_specialization_constants = ash::vk::SpecializationInfo::default()
            .map_entries(&specialization_map_toggle)
            .data(bytemuck::bytes_of(&specialization_constants));

        // Use the input shaders or process the shaders the default.
        let vertex_module = vertex_module
            .unwrap_or_else(|| utils::create_shader_module(device, shaders::BB_TRIANGLE_VERTEX));
        let fragment_module = fragment_module
            .unwrap_or_else(|| utils::create_shader_module(device, shaders::BB_TRIANGLE_FRAGMENT));

        // Define the shader stages that will be used in this pipeline.
        let shader_stages = [
            // Define the vertex shader stage.
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(ENTRY_POINT_MAIN)
                .specialization_info(&vertex_specialization_constants),
            // Define the fragment shader stage.
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(ENTRY_POINT_MAIN),
        ];

        // Using the technique Programmable Vertex Pulling (PVP), specifying vertex input states is somewhat obsolete.
        // However, unless `VK_DYNAMIC_STATE_VERTEX_INPUT_EXT` is enabled, a vertex input state must be specified to the pipeline.
        let vertex_input_create_info = ash::vk::PipelineVertexInputStateCreateInfo::default();

        // Define the input assembly for the pipeline.
        let input_assembly = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST);

        // Dictate that a dynamic viewport and scissor will be used with this pipeline.
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(ash::vk::PolygonMode::FILL)
            .cull_mode(ash::vk::CullModeFlags::BACK)
            .front_face(ash::vk::FrontFace::COUNTER_CLOCKWISE) // This is the default, but making it explicit for clarity.
            .line_width(1.);

        let multisampling = ash::vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(multisample_count.unwrap_or(ash::vk::SampleCountFlags::TYPE_1)) // Must match the sample count of the color attachments of the render pass.
            .min_sample_shading(1.); // Only does something when `sample_shading_enable` is true.

        let color_blend_attachment = [ash::vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false) // TODO: Enable blending for the color attachment.
            .color_write_mask(
                ash::vk::ColorComponentFlags::R
                    | ash::vk::ColorComponentFlags::G
                    | ash::vk::ColorComponentFlags::B
                    | ash::vk::ColorComponentFlags::A,
            )];
        let color_blending = ash::vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment);

        // TODO: Store the push constant ranges with the pipeline so that each can be explicitly reused during the render.
        let push_constant_ranges = [ash::vk::PushConstantRange {
            stage_flags: ash::vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<f32>() as u32,
        }];
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &ash::vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&push_constant_ranges),
                    None,
                )
                .expect("Unable to create the application graphics pipeline layout")
        };

        let depth_stencil_state = ash::vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(true)
            .depth_compare_op(ash::vk::CompareOp::LESS)
            .min_depth_bounds(0.)
            .max_depth_bounds(1.);

        // Assert that the viewport and scissor will be assigned dynamically by the command buffers at render time.
        let dynamic_states = [
            ash::vk::DynamicState::VIEWPORT,
            ash::vk::DynamicState::SCISSOR,
        ];

        // Create the graphics pipeline using the parameters above.
        let pipeline = *unsafe {
            device.create_graphics_pipelines(
                ash::vk::PipelineCache::null(),
                &[ash::vk::GraphicsPipelineCreateInfo {
                    stage_count: shader_stages.len() as u32,
                    p_stages: shader_stages.as_ptr(),
                    p_vertex_input_state: &vertex_input_create_info,
                    p_input_assembly_state: &input_assembly,
                    p_viewport_state: &viewport_state,
                    p_rasterization_state: &rasterizer,
                    p_multisample_state: &multisampling,
                    p_depth_stencil_state: &depth_stencil_state,
                    p_color_blend_state: &color_blending,
                    p_dynamic_state: &ash::vk::PipelineDynamicStateCreateInfo::default()
                        .dynamic_states(&dynamic_states),
                    layout: pipeline_layout,
                    render_pass,
                    subpass: 0, // Optional, but a good reminder that pipelines are associated with a subpass.
                    base_pipeline_index: -1,
                    ..Default::default()
                }],
                None,
            )
        }
        .expect("Unable to create the application graphics pipeline")
        .first()
        .expect("vkCreateGraphicsPipelines returned an empty list of pipelines");

        // Create the framebuffers for this application.
        let framebuffers = create_framebuffers(device, swapchain, render_pass, fxaa_pass);

        Self {
            handle: pipeline,
            layout: pipeline_layout,
            specialization_constants,
            render_pass,
            shaders: Shaders {
                vertex_module,
                fragment_module,
            },
            framebuffers,
        }
    }

    /// Destroy the graphics pipeline and its dependent resources.
    /// # Safety
    /// This function **must** only be called when the owned resources are not currently being processed by the GPU.
    pub fn destroy(self, device: &ash::Device, free_render_pass: bool, free_shaders: bool) {
        unsafe {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);

            if free_render_pass {
                // Destroy the render pass.
                device.destroy_render_pass(self.render_pass, None);
            }
            if free_shaders {
                // Destroy all currently open shader modules.
                device.destroy_shader_module(self.shaders.vertex_module, None);
                device.destroy_shader_module(self.shaders.fragment_module, None);
            }
            for framebuffer in self.framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }
        }
    }

    /// Create the framebuffers, likely after a swapchain recreation.
    /// # Safety
    /// The existing render pass must be valid.
    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        swapchain: &utils::Swapchain,
        fxaa_pass: Option<&utils::fxaa_pass::FxaaPass>,
    ) {
        for framebuffer in &self.framebuffers {
            unsafe {
                device.destroy_framebuffer(*framebuffer, None);
            }
        }
        self.framebuffers = create_framebuffers(device, swapchain, self.render_pass, fxaa_pass);
    }

    /// Recreate the graphics pipeline with updated values.
    /// # Safety
    /// Ensure that the pipeline resources can be safely destroyed at the time of calling this function.
    pub fn recreate(
        &mut self,
        device: &ash::Device,
        create_or_reuse_render_pass: CreateReuseRenderPass,
        swapchain: &utils::Swapchain,
        specialization_constants: SpecializationConstants,
        fxaa_pass: Option<&utils::fxaa_pass::FxaaPass>,
    ) {
        let should_free_old_render_pass = !matches!(&create_or_reuse_render_pass, CreateReuseRenderPass::Reuse(r) if *r == self.render_pass);

        // Create a new graphics pipeline with the updated values.
        let mut new_pipeline = Self::new(
            device,
            Some(self.shaders.vertex_module),
            Some(self.shaders.fragment_module),
            create_or_reuse_render_pass,
            swapchain,
            specialization_constants,
            fxaa_pass,
        );

        // Swap the new graphics pipeline with the old one.
        std::mem::swap(self, &mut new_pipeline);
        let old_pipeline = new_pipeline; // Rename the variable for clarity.

        // Destroy the old graphics pipeline and render pass.
        old_pipeline.destroy(device, should_free_old_render_pass, false);
    }

    /// Render the example triangle to the current frame.
    pub fn render_frame(
        &mut self,
        device: &ash::Device,
        command_buffer: ash::vk::CommandBuffer,
        extent: ash::vk::Extent2D,
        image_index: usize,
        push_constants: &PushConstants,
    ) {
        // Begin the render pass for the current frame.
        unsafe {
            // NOTE: This must match the number of attachments in the render pass, and will be indexed by attachment index.
            let clear_values = [ash::vk::ClearValue::default()];

            device.cmd_begin_render_pass(
                command_buffer,
                &ash::vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.framebuffers[image_index],
                    render_area: ash::vk::Rect2D {
                        offset: ash::vk::Offset2D::default(),
                        extent,
                    },
                    clear_value_count: clear_values.len() as u32,
                    p_clear_values: clear_values.as_ptr(),
                    ..Default::default()
                },
                ash::vk::SubpassContents::INLINE,
            );
        }

        unsafe {
            device.cmd_push_constants(
                command_buffer,
                self.layout,
                ash::vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(push_constants),
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
                self.handle,
            );
        }

        // Draw the example triangle.
        unsafe {
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        // End the render pass for the current frame.
        unsafe {
            device.cmd_end_render_pass(command_buffer);
        }
    }

    // Getters.
    pub fn render_pass(&self) -> ash::vk::RenderPass {
        self.render_pass
    }
    pub fn specialization_constants(&self) -> SpecializationConstants {
        self.specialization_constants
    }
}
