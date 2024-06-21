use std::ffi::CStr;

use smallvec::SmallVec;

pub mod utils;

const EXPECTED_MAX_ATTACHMENTS: usize = 4;
const ENTRY_POINT_MAIN: &'static CStr = c"main";

/// Create the render pass capable of orchestrating the rendering of framebuffers for this application.
pub fn create_render_pass(
    device: &ash::Device,
    swapchain: &utils::Swapchain,
) -> ash::vk::RenderPass {
    let mut color_attachment_references =
        SmallVec::<[ash::vk::AttachmentReference; EXPECTED_MAX_ATTACHMENTS]>::new();
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
            .map(|r| r as *const _)
            .unwrap_or(std::ptr::null()),
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
    image_view: &ash::vk::ImageView,
    extent: ash::vk::Extent2D,
) -> ash::vk::Framebuffer {
    let ash::vk::Extent2D { width, height } = extent;
    unsafe {
        device.create_framebuffer(
            &ash::vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: image_view,
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

/// Create the graphics pipeline capable of rendering this application's scene.
pub fn create_graphics_pipeline(
    swapchain: &utils::Swapchain,
    device: &ash::Device,
    vertex_module: ash::vk::ShaderModule,
    fragment_module: ash::vk::ShaderModule,
    render_pass: ash::vk::RenderPass,
) -> ash::vk::Pipeline {
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

    // Define the viewport rectangle that this pipeline will render to.
    let extent = swapchain.extent();
    let viewport = ash::vk::Viewport {
        x: 0.,
        y: 0.,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.,
        max_depth: 1.,
    };
    let scissor = ash::vk::Rect2D {
        offset: ash::vk::Offset2D::default(),
        extent,
    };
    let viewport_state = ash::vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor,
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

    // TODO: Enable other common pipeline features including descriptor sets, dynamic state, etc.

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

    *unsafe {
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
                p_dynamic_state: &ash::vk::PipelineDynamicStateCreateInfo::default(), // TODO: Allow for the extent to have a dynamic state.
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
    .expect("vkCreateGraphicsPipelines returned an empty list of pipelines")
}
