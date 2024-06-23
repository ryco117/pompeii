use std::ffi::CStr;

use smallvec::SmallVec;

pub mod utils;

/// Store the SPIR-V representation of the shaders in the binary.
mod shaders {
    pub const VERTEX: &[u32] =
        inline_spirv::include_spirv!("src/shaders/bb_triangle_vert.glsl", vert, glsl);
    pub const FRAGMENT: &[u32] =
        inline_spirv::include_spirv!("src/shaders/bb_triangle_frag.glsl", frag, glsl);
}

/// A sane maximum number of color attachments that can be used before requiring a heap allocation.
const EXPECTED_MAX_COLOR_ATTACHMENTS: usize = 4;

/// The default entry point for shaders is the `main` function.
const ENTRY_POINT_MAIN: &CStr = c"main";

/// Create the render pass capable of orchestrating the rendering of framebuffers for this application.
pub fn create_render_pass(
    device: &ash::Device,
    swapchain: &utils::Swapchain,
) -> ash::vk::RenderPass {
    let mut color_attachment_references =
        SmallVec::<[ash::vk::AttachmentReference; EXPECTED_MAX_COLOR_ATTACHMENTS]>::new();
    let mut depth_stencil_attachment_reference = None;

    // TODO: Allow for a list of attachment images with different formats, layouts, etc.
    let format = swapchain.image_format();
    let is_stencil = utils::is_stencil_format(format);
    let is_depth = utils::is_depth_format(format);

    let attachment_descriptor = ash::vk::AttachmentDescription {
        format,
        samples: ash::vk::SampleCountFlags::TYPE_1,
        load_op: ash::vk::AttachmentLoadOp::CLEAR, // Clear the framebuffer before rendering.
        store_op: ash::vk::AttachmentStoreOp::STORE, // Store the framebuffer after rendering.
        stencil_load_op: if is_stencil {
            ash::vk::AttachmentLoadOp::CLEAR // Match the `load_op` above when the format is stencil.
        } else {
            ash::vk::AttachmentLoadOp::DONT_CARE
        },
        stencil_store_op: if is_stencil {
            ash::vk::AttachmentStoreOp::STORE // Match the `store_op` above when the format is stencil.
        } else {
            ash::vk::AttachmentStoreOp::DONT_CARE
        },
        initial_layout: ash::vk::ImageLayout::UNDEFINED,
        final_layout: ash::vk::ImageLayout::PRESENT_SRC_KHR,
        ..Default::default()
    };

    if is_stencil || is_depth {
        depth_stencil_attachment_reference = Some(ash::vk::AttachmentReference {
            attachment: 0, // TODO: Use the index of color attachment in framebuffer.
            layout: ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        });
    } else {
        color_attachment_references.push(ash::vk::AttachmentReference {
            attachment: 0, // TODO: Use the index of color attachment in framebuffer.
            layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        });
    }

    // Define the single subpass that will be used in the render pass.
    let subpass_description = ash::vk::SubpassDescription {
        pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: color_attachment_references.len() as u32,
        p_color_attachments: color_attachment_references.as_ptr(),
        // TODO: Support explicit `p_resolve_attachments` usage.
        p_depth_stencil_attachment: depth_stencil_attachment_reference
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref),
        ..Default::default()
    };

    // TODO: Be more judicious about the flags we set for any particular application's needs.
    let subpass_dependencies = [
        ash::vk::SubpassDependency {
            src_subpass: ash::vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: ash::vk::AccessFlags::MEMORY_READ,
            dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_READ
                | ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::SHADER_READ,
            ..Default::default()
        },
        ash::vk::SubpassDependency {
            src_subpass: 0,
            dst_subpass: ash::vk::SUBPASS_EXTERNAL,
            src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            dst_stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            src_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_READ
                | ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_READ
                | ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::SHADER_READ,
            ..Default::default()
        },
    ];

    unsafe {
        device.create_render_pass(
            &ash::vk::RenderPassCreateInfo {
                attachment_count: 1, // TODO: Support multiple attachments.
                p_attachments: &attachment_descriptor,
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
    image_view: ash::vk::ImageView,
    extent: ash::vk::Extent2D,
) -> ash::vk::Framebuffer {
    let ash::vk::Extent2D { width, height } = extent;
    unsafe {
        device.create_framebuffer(
            &ash::vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: &image_view,
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

/// Manage the graphics pipeline and dependent resources.
pub struct Pipeline {
    pub handle: ash::vk::Pipeline,
    pub layout: ash::vk::PipelineLayout,
}

impl Pipeline {
    /// Create the graphics pipeline capable of rendering this application's scene.
    pub fn new_graphics(
        device: &ash::Device,
        vertex_module: ash::vk::ShaderModule,
        fragment_module: ash::vk::ShaderModule,
        render_pass: ash::vk::RenderPass,
    ) -> Self {
        let shader_stages = [
            // Define the vertex shader stage.
            ash::vk::PipelineShaderStageCreateInfo {
                stage: ash::vk::ShaderStageFlags::VERTEX,
                module: vertex_module,
                p_name: ENTRY_POINT_MAIN.as_ptr(),
                ..Default::default()
            },
            // Define the fragment shader stage.
            ash::vk::PipelineShaderStageCreateInfo {
                stage: ash::vk::ShaderStageFlags::FRAGMENT,
                module: fragment_module,
                p_name: ENTRY_POINT_MAIN.as_ptr(),
                ..Default::default()
            },
        ];

        // TODO: Allow for a list of vertex input bindings and attributes.
        let vertex_input_create_info = ash::vk::PipelineVertexInputStateCreateInfo {
            ..Default::default()
        };

        let input_assembly = ash::vk::PipelineInputAssemblyStateCreateInfo {
            topology: ash::vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        // Define a dynamic viewport and scissor will be used with this pipeline.
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: ash::vk::PolygonMode::FILL, // TODO: Allow the caller to define the fill mode.
            cull_mode: ash::vk::CullModeFlags::BACK,
            front_face: ash::vk::FrontFace::COUNTER_CLOCKWISE, // This is the default, but making it explicit for clarity.
            line_width: 1.,
            ..Default::default()
        };

        // TODO: Allow for the rasterizer to apply multisampling.
        let multisampling = ash::vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: ash::vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 1., // Only does something when `sample_shading_enable` is true.
            ..Default::default()
        };

        // TODO: Enable blending for the color attachment.
        let color_blend_attachment = ash::vk::PipelineColorBlendAttachmentState {
            color_write_mask: ash::vk::ColorComponentFlags::R
                | ash::vk::ColorComponentFlags::G
                | ash::vk::ColorComponentFlags::B
                | ash::vk::ColorComponentFlags::A,
            ..Default::default()
        };
        let color_blending = ash::vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: ash::vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        // TODO: Enable other common pipeline features including descriptor sets, push-constant ranges, etc.

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&ash::vk::PipelineLayoutCreateInfo::default(), None)
                .expect("Unable to create the application graphics pipeline layout")
        };

        let depth_stencil_state = ash::vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: ash::vk::FALSE,
            depth_write_enable: ash::vk::TRUE,
            depth_compare_op: ash::vk::CompareOp::LESS,
            min_depth_bounds: 0.,
            max_depth_bounds: 1.,
            ..Default::default()
        };

        // Only used with dynamic rendering with field `p_next`. TODO: Add remaining fields when allowing dynamic rendering.
        // let pipeline_rendering_create_info = ash::vk::PipelineRenderingCreateInfo {
        //     color_attachment_count: 1,
        //     p_color_attachment_formats: &swapchain.image_format(),
        //     ..Default::default()
        // };

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
                    p_dynamic_state: &ash::vk::PipelineDynamicStateCreateInfo {
                        dynamic_state_count: dynamic_states.len() as u32,
                        p_dynamic_states: dynamic_states.as_ptr(),
                        ..Default::default()
                    },
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

        Self {
            handle: pipeline,
            layout: pipeline_layout,
        }
    }

    /// Destroy the graphics pipeline and its dependent resources.
    /// # Safety
    /// This function **must** only be called when the owned resources are not currently being processed by the GPU.
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Define which rendering objects are necessary for this application.
pub struct Renderer {
    pub physical_device: ash::vk::PhysicalDevice,
    pub logical_device: ash::Device,
    pub surface: ash::vk::SurfaceKHR,
    pub swapchain: utils::Swapchain,
    pub render_pass: ash::vk::RenderPass,
    pub shaders: Vec<ash::vk::ShaderModule>,
    pub graphics_pipeline: Pipeline,
    pub graphics_queue: ash::vk::Queue,
    pub presentation_queue: ash::vk::Queue,
    pub command_pool: ash::vk::CommandPool,
    pub command_buffers: Vec<ash::vk::CommandBuffer>,
    pub framebuffers: Vec<ash::vk::Framebuffer>,
    pub fences_and_state: Vec<(ash::vk::Fence, bool)>,

    pub swapchain_preferences: utils::SwapchainPreferences,
}

impl Renderer {
    /// Create a new renderer for the application.
    pub fn new(
        vulkan: &utils::VulkanCore,
        surface: ash::vk::SurfaceKHR,
        swapchain_preferences: utils::SwapchainPreferences,
    ) -> Self {
        // Required device extensions for the swapchain.
        const DEVICE_EXTENSIONS: [*const i8; 1] = [ash::khr::swapchain::NAME.as_ptr()];

        // Use simple heuristics to find the best suitable physical device.
        let (physical_device, device_properties) = {
            let devices = utils::get_sorted_physical_devices(&vulkan.instance, &DEVICE_EXTENSIONS);

            *devices
                .first()
                .expect("Unable to find a suitable physical device")
        };

        #[cfg(debug_assertions)]
        println!(
            "Selected physical device: {:?}",
            device_properties
                .device_name_as_c_str()
                .expect("Unable to get device name")
        );

        // Create a logical device capable of rendering to the surface and performing compute operations.
        // TODO: Support having a list of queue families for each type for greater flexibility.
        let (logical_device, queue_families) =
            utils::new_device(vulkan, physical_device, surface, &DEVICE_EXTENSIONS);
        let utils::QueueFamilies {
            graphics: Some(graphics_index),
            present: Some(present_index),
            ..
        } = queue_families
        else {
            panic!("Unable to find suitable queue families");
        };

        // Create a queue capable of performing graphics commands.
        let graphics_queue = unsafe { logical_device.get_device_queue(graphics_index, 0) };

        // Create a presentation queue for use with the swapchain.
        let presentation_queue = unsafe { logical_device.get_device_queue(present_index, 0) };

        // Create an object to manage the swapchain, its images, and synchronization primitives.
        let swapchain = utils::Swapchain::new(
            vulkan,
            physical_device,
            &logical_device,
            surface,
            swapchain_preferences,
            None,
        );
        let image_count = swapchain.image_count();

        // Create the render pass that will orchestrate usage of the attachments in a framebuffer's render.
        let render_pass = create_render_pass(&logical_device, &swapchain);

        // Process the shaders that will be used in the graphics pipeline.
        let vertex_module = utils::create_shader_module(&logical_device, shaders::VERTEX);
        let fragment_module = utils::create_shader_module(&logical_device, shaders::FRAGMENT);

        // Create the graphics pipeline that will be used to render the application.
        let graphics_pipeline =
            Pipeline::new_graphics(&logical_device, vertex_module, fragment_module, render_pass);

        // Create a pool for allocating new commands.
        // NOTE: https://developer.nvidia.com/blog/vulkan-dos-donts/ Recommends `image_count * recording_thread_count` many command pools for optimal command buffer allocation.
        //       However, we currently only reuse existing command buffers and do not need to allocate new ones.
        let command_pool = unsafe {
            logical_device
                .create_command_pool(
                    &ash::vk::CommandPoolCreateInfo {
                        flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                        queue_family_index: graphics_index,
                        ..Default::default()
                    },
                    None,
                )
                .expect("Unable to create command pool")
        };

        // Allocate a command buffer for each image in the swapchain.
        // One may want to use a different number if there are background tasks not related to an image.
        let command_buffer_info = ash::vk::CommandBufferAllocateInfo {
            command_pool,
            level: ash::vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffers = unsafe {
            let mut c = Vec::new();
            c.resize_with(image_count, || {
                *logical_device
                    .allocate_command_buffers(&command_buffer_info)
                    .expect("Unable to allocate command buffer")
                    .first()
                    .expect("No command buffers were allocated")
            });
            c
        };

        // Create a fence for each image in the swapchain so the CPU can wait for the GPU to finish a given frame.
        let fence_create_info = ash::vk::FenceCreateInfo {
            flags: ash::vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let fences_and_state = unsafe {
            let mut f = Vec::new();
            f.resize_with(image_count, || {
                let f = logical_device
                    .create_fence(&fence_create_info, None)
                    .expect("Unable to create fence");
                (f, false /* is_submitted */)
            });
            f
        };

        // Create the framebuffers for this application.
        // TODO: Create framebuffers in the same function as the render pass because the render pass defines the framebuffer topology.
        let extent = swapchain.extent();
        let framebuffers = swapchain
            .image_views()
            .iter()
            .map(|image_view| create_framebuffer(&logical_device, render_pass, *image_view, extent))
            .collect();

        Self {
            physical_device,
            logical_device,
            surface,
            swapchain,
            render_pass,
            shaders: vec![vertex_module, fragment_module],
            graphics_pipeline,
            graphics_queue,
            presentation_queue,
            command_pool,
            command_buffers,
            framebuffers,
            fences_and_state,
            swapchain_preferences,
        }
    }

    /// Destroy the Pompeii renderer and its dependent resources.
    /// # Safety
    /// This function **must** only be called when the owned resources are not currently being processed by the GPU.
    pub fn destroy(mut self) {
        unsafe {
            for (fence, _) in self.fences_and_state {
                self.logical_device.destroy_fence(fence, None);
            }

            self.logical_device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.logical_device
                .destroy_command_pool(self.command_pool, None);

            self.graphics_pipeline.destroy(&self.logical_device);
            for shader in self.shaders {
                self.logical_device.destroy_shader_module(shader, None);
            }

            self.logical_device
                .destroy_render_pass(self.render_pass, None);

            self.swapchain
                .destroy(&self.logical_device, &mut self.framebuffers);

            self.logical_device.destroy_device(None);
        }
    }

    /// Recreate the swapchain, including the framebuffers and image views for the frames owned by the swapchain.
    /// The `self.swapchain_preferences` are used to recreate the swapchain and do not need to match those used with the initial swapchain creation.
    pub fn recreate_swapchain(&mut self, vulkan: &utils::VulkanCore) {
        self.swapchain.recreate_swapchain(
            vulkan,
            self.physical_device,
            &self.logical_device,
            self.surface,
            &mut self.framebuffers,
            self.swapchain_preferences,
            |img, extent| create_framebuffer(&self.logical_device, self.render_pass, img, extent),
        );
    }

    /// Attempt to render the next frame of the application. If there is a recoverable error, then the swapchain is recreated and the function bails early without rendering.
    pub fn render_frame(&mut self, vulkan: &utils::VulkanCore) {
        // Get the next image to render to. Has internal synchronization to ensure the previous acquire completed on the GPU.
        let (_image, image_index) = match self.swapchain.acquire_next_image(&self.logical_device) {
            Ok((image, index, false)) => (image, index),

            // TODO: Consider accepting suboptimal for this draw, but set a flag to recreate the swapchain next frame.
            Ok((_, _, true)) | Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let surface_capabilities = unsafe {
                    vulkan
                        .khr
                        .get_physical_device_surface_capabilities(
                            self.physical_device,
                            self.surface,
                        )
                        .expect("Unable to get surface capabilities")
                };
                if surface_capabilities.max_image_extent.width == 0
                    || surface_capabilities.max_image_extent.height == 0
                {
                    println!("WARN: Surface capabilities are zero at image acquire, skipping swapchain recreation");
                    return;
                }

                println!("WARN: Swapchain is out of date at image acquire, needs to be recreated.");
                self.recreate_swapchain(vulkan);
                return;
            }

            Err(e) => panic!("Unable to acquire next image from swapchain: {e}"),
        };

        // Synchronize the CPU with the GPU for the resources previously used for this image index.
        // TODO: Make this into its own utils function because it is a common pattern.
        let resource_fence = self.fences_and_state[image_index as usize].0;
        let command_buffer = self.command_buffers[image_index as usize];
        unsafe {
            self.logical_device
                .wait_for_fences(&[resource_fence], true, utils::FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Unable to wait for fence to begin frame");
            self.logical_device
                .reset_command_buffer(
                    command_buffer,
                    ash::vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Unable to reset command buffer");
            self.logical_device
                .begin_command_buffer(
                    command_buffer,
                    &ash::vk::CommandBufferBeginInfo {
                        flags: ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .expect("Unable to begin command buffer");
        }

        let extent = self.swapchain.extent();

        // Begin the render pass for the current frame.
        unsafe {
            self.logical_device.cmd_begin_render_pass(
                command_buffer,
                &ash::vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.framebuffers[image_index as usize],
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

        // Set the viewport and scissor in the command buffer because we specified they would be set dynamically in the pipeline.
        unsafe {
            self.logical_device.cmd_set_viewport(
                command_buffer,
                0,
                &[ash::vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.logical_device.cmd_set_scissor(
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
            self.logical_device.cmd_bind_pipeline(
                command_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.handle,
            );
        }

        // TODO: Update descriptor sets here when we have them.

        // Draw the example triangle.
        unsafe {
            self.logical_device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        // End the render pass for the current frame.
        unsafe {
            self.logical_device.cmd_end_render_pass(command_buffer);
            self.logical_device
                .end_command_buffer(command_buffer)
                .expect("Unable to end the graphics command buffer");
        }

        // Submit the draw command buffer to the GPU.
        let submit_info = ash::vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.swapchain.image_available(),
            p_wait_dst_stage_mask: &ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 1,
            p_signal_semaphores: &self.swapchain.image_rendered(),
            ..Default::default()
        };
        unsafe {
            self.logical_device
                .reset_fences(&[resource_fence])
                .expect("Unable to reset fence for this frame's resources");
            self.logical_device
                .queue_submit(self.graphics_queue, &[submit_info], resource_fence)
                .expect("Unable to submit command buffer");

            // Note that this fence has been submitted.
            // TODO: Determine if this is useful.
            self.fences_and_state[image_index as usize].1 = true;
        }

        // Queue the presentation of the swapchain image.
        match self.swapchain.present(self.presentation_queue) {
            Ok(false) => (),
            Ok(true) | Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let surface_capabilities = unsafe {
                    vulkan
                        .khr
                        .get_physical_device_surface_capabilities(
                            self.physical_device,
                            self.surface,
                        )
                        .expect("Unable to get surface capabilities")
                };
                if surface_capabilities.max_image_extent.width == 0
                    && surface_capabilities.max_image_extent.height == 0
                {
                    #[cfg(debug_assertions)]
                    println!("Surface capabilities are zero at image presentation, skipping swapchain recreation");
                    return;
                }

                println!("Swapchain is out of date at image presentation, needs to be recreated.");
                self.recreate_swapchain(vulkan);
            }
            Err(e) => panic!("Unable to present swapchain image: {e:?}"),
        }
    }
}
