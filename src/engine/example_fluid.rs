use crate::engine::utils::{
    self,
    shaders::{ENTRY_POINT_MAIN, FULLSCREEN_VERTEX},
    textures::AllocatedImage,
    FIVE_SECONDS_IN_NANOSECONDS,
};

pub mod shaders {
    /// Standard triangle-example fragment shader.
    pub const FLUID_ADVECTION: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid/advection.comp", comp, glsl);
    pub const FLUID_CURL: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid/curl.comp", comp, glsl);
    pub const FLUID_DIVERGENCE: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid/divergence.comp", comp, glsl);
    pub const FLUID_GRADIENT_SUBTRACT: &[u32] = inline_spirv::include_spirv!(
        "src/shaders/example_fluid/gradient_subtract.comp",
        comp,
        glsl
    );
    pub const FLUID_PRESSURE: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid/pressure.comp", comp, glsl);
    pub const FLUID_VORTICITY: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid/vorticity.comp", comp, glsl);
    pub const FLUID_FRAGMENT: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid.frag", frag, glsl);
}

/// The constant used to define the number of iterations used to adjust the pressure towards a divergence-free field.
/// This value is half the number of iterations used elsewhere because each iteration has two stages which are interleaved.
const MAX_PRESSURE_SMOOTHING_ITERATIONS: u32 = 16;

/// Define the shared push constants for each compute stage of this minimal fluid simulation.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
pub struct ComputePushConstants {
    // Fluid simulation parameters.
    pub cursor_dye: [f32; 4],
    pub cursor_position: [f32; 2],
    pub cursor_velocity: [f32; 2],
    pub screen_size: [u32; 2],

    pub delta_time: f32,

    pub velocity_diffusion_rate: f32,
    pub dye_diffusion_rate: f32,

    /// Sane values are 0 to 50. Default is 30.
    pub vorticity_strength: f32,
}

/// Define the texture to display from the fluid simulation.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, strum::EnumCount, strum::FromRepr, bytemuck::NoUninit)]
pub enum FluidDisplayTexture {
    #[default]
    Velocity,
    Dye,
    Pressure,
    ColorFieldNormal,
    ColorFieldLine,
    ColorFieldCircle,
    ColorFieldGradient,
}
impl FluidDisplayTexture {
    /// Get the next fluid display variant in a cycle of types.
    pub fn next(self) -> Self {
        FluidDisplayTexture::from_repr((self as u32).wrapping_add(1))
            .unwrap_or(FluidDisplayTexture::Velocity)
    }
}

/// Use a separate set of push constants to choose how to render the output of the fluid simulation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FragmentPushConstants {
    // Fluid simulation parameters.
    pub screen_size: [u32; 2],
    pub display_texture: FluidDisplayTexture,
}

/// Create the render pass capable of orchestrating the rendering of framebuffers for this application.
fn create_render_pass(
    device: &ash::Device,
    image_format: ash::vk::Format,
    destination_layout: ash::vk::ImageLayout,
) -> ash::vk::RenderPass {
    // Create the attachment descriptor defining the color attachment output from the fragment shader.
    let color_attachment_descriptor = ash::vk::AttachmentDescription {
        format: image_format,
        samples: ash::vk::SampleCountFlags::TYPE_1, // This setup has nothing to gain from multisampling.
        load_op: ash::vk::AttachmentLoadOp::DONT_CARE, // Each pixel will be rewritten based on the input images.
        store_op: ash::vk::AttachmentStoreOp::STORE,   // Store the framebuffer after rendering.
        stencil_load_op: ash::vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: ash::vk::ImageLayout::UNDEFINED,
        final_layout: destination_layout,
        ..Default::default()
    };

    // Define the index the subpass will use to define their color attachment.
    let color_attachment_reference = ash::vk::AttachmentReference {
        attachment: 0, // NOTE: Uses the index of color attachment in framebuffer.
        layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    // Define the single subpass that will be used in the render pass.
    let subpass_description = ash::vk::SubpassDescription {
        pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &color_attachment_reference,
        ..Default::default()
    };

    // Make explicit the dependency the fragment shader has on the write output of the compute shader stages.
    let subpass_dependencies = [
        ash::vk::SubpassDependency {
            src_subpass: ash::vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: ash::vk::AccessFlags::SHADER_READ,
            ..Default::default()
        },
        ash::vk::SubpassDependency {
            src_subpass: ash::vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        },
    ];

    unsafe {
        device.create_render_pass(
            &ash::vk::RenderPassCreateInfo {
                attachment_count: 1,
                p_attachments: &color_attachment_descriptor,
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

fn initialize_image_layouts(
    device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    images: &[AllocatedImage],
    layout: ash::vk::ImageLayout,
) -> ash::vk::Fence {
    let command_buffer = unsafe {
        device.allocate_command_buffers(
            &ash::vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(ash::vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("Failed to allocate command buffer for image layout initialization")[0];

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }
    .expect("Failed to begin command buffer for image layout initialization");

    let image_barriers = images
        .iter()
        .map(|image| {
            ash::vk::ImageMemoryBarrier2::default()
                .src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
                .src_access_mask(ash::vk::AccessFlags2::NONE)
                .dst_stage_mask(
                    ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                        | ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .dst_access_mask(
                    ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | ash::vk::AccessFlags2::SHADER_WRITE,
                )
                .old_layout(ash::vk::ImageLayout::UNDEFINED)
                .new_layout(layout)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        })
        .collect::<Vec<_>>();
    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &ash::vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
        );
    }

    unsafe { device.end_command_buffer(command_buffer) }
        .expect("Failed to end command buffer for image layout initialization");

    unsafe {
        let fence = device
            .create_fence(&ash::vk::FenceCreateInfo::default(), None)
            .expect("Failed to create fence for image layout initialization");

        device
            .queue_submit(
                queue,
                &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                fence,
            )
            .expect("Failed to submit command buffer for image layout initialization");

        fence
    }
}

/// Create a framebuffer that can be used with this render pass.
fn create_framebuffers(
    device: &ash::Device,
    memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    extent: ash::vk::Extent2D,
    destination_views: &[ash::vk::ImageView],
    render_pass: ash::vk::RenderPass,
) -> (Vec<ash::vk::Framebuffer>, [AllocatedImage; 8]) {
    // Create several images for storing the partial results of the fluid simulation each frame.
    // These images are not strictly part of the framebuffers, but are used in the fluid simulation and dependent on the size of the surface.
    let image_info = ash::vk::ImageCreateInfo::default()
        .image_type(ash::vk::ImageType::TYPE_2D)
        .extent(ash::vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(ash::vk::SampleCountFlags::TYPE_1)
        .usage(ash::vk::ImageUsageFlags::STORAGE);

    let input_velocity_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32G32_SFLOAT),
        None,
        "Fluid Sim input velocity buffer",
    );
    let curl_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32_SFLOAT),
        None,
        "Fluid Sim curl buffer",
    );
    let divergence_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32_SFLOAT),
        None,
        "Fluid Sim divergence buffer",
    );
    let alpha_pressure_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32_SFLOAT),
        None,
        "Fluid Sim alpha pressure buffer",
    );
    let beta_pressure_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32_SFLOAT),
        None,
        "Fluid Sim beta pressure buffer",
    );
    let output_velocity_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32G32_SFLOAT),
        None,
        "Fluid Sim output velocity buffer",
    );
    let input_dye_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32G32B32A32_SFLOAT),
        None,
        "Fluid Sim input dye buffer",
    );
    let output_dye_image = AllocatedImage::new(
        device,
        memory_allocator,
        image_info.format(ash::vk::Format::R32G32B32A32_SFLOAT),
        None,
        "Fluid Sim output dye buffer",
    );

    let images = [
        input_velocity_image,
        curl_image,
        divergence_image,
        alpha_pressure_image,
        beta_pressure_image,
        output_velocity_image,
        input_dye_image,
        output_dye_image,
    ];

    let fence = initialize_image_layouts(
        device,
        command_pool,
        queue,
        &images,
        ash::vk::ImageLayout::GENERAL,
    );

    // The actual render pass framebuffers simply draw to the destination views as color attachments.
    let framebuffers = destination_views
        .iter()
        .map(|destination_view| {
            let framebuffer_info = ash::vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: destination_view,
                width: extent.width,
                height: extent.height,
                layers: 1,
                ..Default::default()
            };
            unsafe { device.create_framebuffer(&framebuffer_info, None) }
                .expect("Failed to create framebuffer for FXAA post-processing")
        })
        .collect();

    unsafe {
        device
            .wait_for_fences(&[fence], true, FIVE_SECONDS_IN_NANOSECONDS)
            .expect("Failed to wait for fence to signal for image layout initialization");
        device.destroy_fence(fence, None);
    }

    (framebuffers, images)
}

fn create_descriptor_set_layouts(device: &ash::Device) -> [ash::vk::DescriptorSetLayout; 2] {
    let storage_binding_base = ash::vk::DescriptorSetLayoutBinding::default()
        .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
    let input_velocity = storage_binding_base.binding(0);
    let curl_texture = storage_binding_base.binding(1);
    let divergence_texture = storage_binding_base.binding(2);
    let alpha_pressure_texture = storage_binding_base.binding(3);
    let beta_pressure_texture = storage_binding_base.binding(4);
    let output_velocity = storage_binding_base.binding(5);
    let input_dye = storage_binding_base.binding(6);
    let output_dye = storage_binding_base.binding(7);

    let compute = unsafe {
        device.create_descriptor_set_layout(
            &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                input_velocity,
                curl_texture,
                divergence_texture,
                alpha_pressure_texture,
                beta_pressure_texture,
                output_velocity,
                input_dye,
                output_dye,
            ]),
            None,
        )
    }
    .expect("Unable to create the compute descriptor set layout for the fluid simulation");

    let input_velocity = storage_binding_base
        .binding(0)
        .stage_flags(ash::vk::ShaderStageFlags::FRAGMENT);
    let input_dye = storage_binding_base
        .binding(1)
        .stage_flags(ash::vk::ShaderStageFlags::FRAGMENT);
    let pressure_texture = storage_binding_base
        .binding(2)
        .stage_flags(ash::vk::ShaderStageFlags::FRAGMENT);
    let graphics = unsafe {
        device.create_descriptor_set_layout(
            &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                input_velocity,
                input_dye,
                pressure_texture,
            ]),
            None,
        )
    }
    .expect("Unable to create the graphics descriptor set layout for the fluid simulation");

    [compute, graphics]
}

/// Create the compute and graphics pipeline layouts.
fn create_pipeline_layout(
    device: &ash::Device,
    compute_descriptor_set_layout: ash::vk::DescriptorSetLayout,
    graphics_descriptor_set_layout: ash::vk::DescriptorSetLayout,
) -> [ash::vk::PipelineLayout; 2] {
    // Create the push constant specification for the fluid simulation. The same data is used for both the compute and fragment shaders.
    let compute_push_constant_range = ash::vk::PushConstantRange {
        stage_flags: ash::vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: std::mem::size_of::<ComputePushConstants>() as u32,
    };

    // Create the pipeline layout for the fluid simulation.
    let compute_pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &ash::vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[compute_descriptor_set_layout])
                .push_constant_ranges(&[compute_push_constant_range]),
            None,
        )
    }
    .expect("Unable to create the compute pipeline layout for the fluid simulation");

    let graphics_push_constant_range = ash::vk::PushConstantRange {
        stage_flags: ash::vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: std::mem::size_of::<FragmentPushConstants>() as u32,
    };
    let graphics_pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &ash::vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[graphics_descriptor_set_layout])
                .push_constant_ranges(&[graphics_push_constant_range]),
            None,
        )
    }
    .expect("Unable to create the graphics pipeline layout for the fluid simulation");

    [compute_pipeline_layout, graphics_pipeline_layout]
}

/// Helper for creating and maintaining the shaders used in the fluid simulation.
struct FluidShaders {
    pub advection: ash::vk::ShaderModule,
    pub curl: ash::vk::ShaderModule,
    pub divergence: ash::vk::ShaderModule,
    pub gradient_subtract: ash::vk::ShaderModule,
    pub pressure: ash::vk::ShaderModule,
    pub vorticity: ash::vk::ShaderModule,
    pub fragment: ash::vk::ShaderModule,
    pub vertex: ash::vk::ShaderModule,
}
impl FluidShaders {
    /// Create the shader modules for the fluid simulation.
    pub fn new(device: &ash::Device) -> Self {
        let advection = utils::create_shader_module(device, shaders::FLUID_ADVECTION);
        let curl = utils::create_shader_module(device, shaders::FLUID_CURL);
        let divergence = utils::create_shader_module(device, shaders::FLUID_DIVERGENCE);
        let gradient_subtract =
            utils::create_shader_module(device, shaders::FLUID_GRADIENT_SUBTRACT);
        let pressure = utils::create_shader_module(device, shaders::FLUID_PRESSURE);
        let vorticity = utils::create_shader_module(device, shaders::FLUID_VORTICITY);
        let fragment = utils::create_shader_module(device, shaders::FLUID_FRAGMENT);
        let vertex = utils::create_shader_module(device, FULLSCREEN_VERTEX);

        Self {
            advection,
            curl,
            divergence,
            gradient_subtract,
            pressure,
            vorticity,
            fragment,
            vertex,
        }
    }

    /// Destroy the shader modules for the fluid simulation.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_shader_module(self.advection, None);
            device.destroy_shader_module(self.curl, None);
            device.destroy_shader_module(self.divergence, None);
            device.destroy_shader_module(self.gradient_subtract, None);
            device.destroy_shader_module(self.pressure, None);
            device.destroy_shader_module(self.vorticity, None);
            device.destroy_shader_module(self.fragment, None);
            device.destroy_shader_module(self.vertex, None);
        }
    }
}

/// Helper for creating compute pipelines for the fluid simulation. The simulation requires multiple synchronous compute pipelines.
fn create_compute_pipeline(
    device: &ash::Device,
    pipeline_layout: ash::vk::PipelineLayout,
    shader_module: ash::vk::ShaderModule,
    specialization_constants: Option<ash::vk::SpecializationInfo>,
) -> ash::vk::Pipeline {
    let mut shader_stage_create_info = ash::vk::PipelineShaderStageCreateInfo::default()
        .stage(ash::vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(ENTRY_POINT_MAIN);
    if let Some(specialization_constants) = &specialization_constants {
        shader_stage_create_info =
            shader_stage_create_info.specialization_info(specialization_constants);
    }

    unsafe {
        device.create_compute_pipelines(
            ash::vk::PipelineCache::null(),
            &[ash::vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage_create_info)
                .layout(pipeline_layout)],
            None,
        )
    }.expect("Unable to create the compute pipeline for the fluid simulation")
        .into_iter().next().expect("vkCreateComputePipelines returned an empty list of pipelines but provided a successful result")
}

/// Helper to create and manage all of the compute pipelines for the fluid simulation.
struct FluidComputeStages {
    advection: ash::vk::Pipeline,
    curl: ash::vk::Pipeline,
    divergence: ash::vk::Pipeline,
    gradient_subtract: ash::vk::Pipeline,
    alpha_pressure: ash::vk::Pipeline,
    beta_pressure: ash::vk::Pipeline,
    vorticity: ash::vk::Pipeline,
}
impl FluidComputeStages {
    /// Helper to create all of the compute pipelines for the fluid simulation.
    pub fn new(
        device: &ash::Device,
        pipeline_layout: ash::vk::PipelineLayout,
        shaders: &FluidShaders,
    ) -> Self {
        let advection = create_compute_pipeline(device, pipeline_layout, shaders.advection, None);
        let curl = create_compute_pipeline(device, pipeline_layout, shaders.curl, None);
        let divergence = create_compute_pipeline(device, pipeline_layout, shaders.divergence, None);
        let gradient_subtract =
            create_compute_pipeline(device, pipeline_layout, shaders.gradient_subtract, None);
        let pressure_specialization_map = ash::vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: std::mem::size_of::<u32>(),
        };
        let alpha_pressure = create_compute_pipeline(
            device,
            pipeline_layout,
            shaders.pressure,
            Some(
                ash::vk::SpecializationInfo::default()
                    .map_entries(&[pressure_specialization_map])
                    .data(&u32::to_ne_bytes(ash::vk::TRUE)),
            ),
        );
        let beta_pressure = create_compute_pipeline(
            device,
            pipeline_layout,
            shaders.pressure,
            Some(
                ash::vk::SpecializationInfo::default()
                    .map_entries(&[pressure_specialization_map])
                    .data(&u32::to_ne_bytes(ash::vk::FALSE)),
            ),
        );
        let vorticity = create_compute_pipeline(device, pipeline_layout, shaders.vorticity, None);

        Self {
            advection,
            curl,
            divergence,
            gradient_subtract,
            alpha_pressure,
            beta_pressure,
            vorticity,
        }
    }

    /// Destroy all of the compute pipelines for the fluid simulation.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.advection, None);
            device.destroy_pipeline(self.curl, None);
            device.destroy_pipeline(self.divergence, None);
            device.destroy_pipeline(self.gradient_subtract, None);
            device.destroy_pipeline(self.alpha_pressure, None);
            device.destroy_pipeline(self.beta_pressure, None);
            device.destroy_pipeline(self.vorticity, None);
        }
    }
}

/// Create the graphics pipeline for the fluid simulation.
fn create_graphics_pipeline(
    device: &ash::Device,
    shaders: &FluidShaders,
    pipeline_layout: ash::vk::PipelineLayout,
    render_pass: ash::vk::RenderPass,
) -> ash::vk::Pipeline {
    let shader_stages = [
        ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::VERTEX)
            .module(shaders.vertex)
            .name(ENTRY_POINT_MAIN),
        ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::FRAGMENT)
            .module(shaders.fragment)
            .name(ENTRY_POINT_MAIN),
    ];

    let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(ash::vk::PolygonMode::FILL)
        .cull_mode(ash::vk::CullModeFlags::NONE)
        .line_width(1.);

    let multisampling = ash::vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(ash::vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.); // Only does something when `sample_shading_enable` is true.

    let color_blend_attachments = [ash::vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(
            ash::vk::ColorComponentFlags::R
                | ash::vk::ColorComponentFlags::G
                | ash::vk::ColorComponentFlags::B
                | ash::vk::ColorComponentFlags::A,
        )];
    let color_blending =
        ash::vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

    unsafe {
        device.create_graphics_pipelines(
            ash::vk::PipelineCache::null(),
            &[ash::vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&ash::vk::PipelineVertexInputStateCreateInfo::default())
                .input_assembly_state(&ash::vk::PipelineInputAssemblyStateCreateInfo::default().topology(ash::vk::PrimitiveTopology::TRIANGLE_STRIP))
                .viewport_state(&ash::vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1))
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .color_blend_state(&color_blending)
                .dynamic_state(&ash::vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                    ash::vk::DynamicState::VIEWPORT,
                    ash::vk::DynamicState::SCISSOR,
                ]))
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .base_pipeline_index(-1)],
            None,
        )
    }.expect("Unable to create the graphics pipeline for the fluid simulation")
        .into_iter().next().expect("vkCreateGraphicsPipelines returned an empty list of pipelines but provided a successful result")
}

fn update_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[ash::vk::DescriptorSet],
    images: &[AllocatedImage],
) {
    let image_infos = images
        .iter()
        .map(|img| {
            ash::vk::DescriptorImageInfo::default()
                .image_view(img.image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL)
        })
        .collect::<Vec<_>>();
    let compute_writes = image_infos.iter().enumerate().map(|img| {
        ash::vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[0])
            .dst_binding(img.0 as u32)
            .descriptor_count(1)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(img.1))
    });

    let graphics_writes = [
        ash::vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[1])
            .dst_binding(0)
            .descriptor_count(1)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(image_infos.get(5).unwrap())),
        ash::vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[1])
            .dst_binding(1)
            .descriptor_count(1)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(image_infos.get(7).unwrap())),
        ash::vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[1])
            .dst_binding(2)
            .descriptor_count(1)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(image_infos.get(3).unwrap())),
    ];

    let writes = compute_writes
        .chain(graphics_writes.into_iter())
        .collect::<Vec<_>>();

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn allocate_descriptor_sets(
    device: &ash::Device,
    descriptor_set_layouts: &[ash::vk::DescriptorSetLayout],
) -> (ash::vk::DescriptorPool, Vec<ash::vk::DescriptorSet>) {
    // NOTE: There are 8 storage images in the compute pipelines, and 3 in the graphics pipeline.
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(2)
                .pool_sizes(&[ash::vk::DescriptorPoolSize::default()
                    .ty(ash::vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(11)]),
            None,
        )
    }
    .expect("Unable to create the descriptor pool for the fluid simulation");

    let allocate_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(descriptor_set_layouts);
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .expect("Unable to allocate descriptor sets for the fluid simulation");

    (descriptor_pool, descriptor_sets)
}

/// The fluid simulation renderer and resources.
pub struct FluidSimulation {
    shaders: FluidShaders,
    compute_descriptor_set_layout: ash::vk::DescriptorSetLayout,
    graphics_descriptor_set_layout: ash::vk::DescriptorSetLayout,
    compute_pipeline_layout: ash::vk::PipelineLayout,
    graphics_pipeline_layout: ash::vk::PipelineLayout,
    render_pass: ash::vk::RenderPass,
    compute_pipelines: FluidComputeStages,
    graphics_pipeline: ash::vk::Pipeline,
    framebuffers: Vec<ash::vk::Framebuffer>,
    allocated_images: Vec<AllocatedImage>,
    compute_fence: ash::vk::Fence,
    graphics_fence: Option<ash::vk::Fence>,
    compute_command_buffer: ash::vk::CommandBuffer,
    descriptor_pool: ash::vk::DescriptorPool,
    descriptor_sets: Vec<ash::vk::DescriptorSet>,
    current_display_texture: FluidDisplayTexture,
}
impl FluidSimulation {
    /// Create a new fluid simulation renderer from the swapchain image properties.
    pub fn new(
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
        extent: ash::vk::Extent2D,
        image_format: ash::vk::Format,
        destination_layout: ash::vk::ImageLayout,
        destination_views: &[ash::vk::ImageView],
        compute_command_pool: ash::vk::CommandPool,
        compute_queue: ash::vk::Queue,
    ) -> Self {
        let shaders = FluidShaders::new(device);
        let [compute_descriptor_set_layout, graphics_descriptor_set_layout] =
            create_descriptor_set_layouts(device);
        let [compute_pipeline_layout, graphics_pipeline_layout] = create_pipeline_layout(
            device,
            compute_descriptor_set_layout,
            graphics_descriptor_set_layout,
        );

        let render_pass = create_render_pass(device, image_format, destination_layout);

        let compute_pipelines = FluidComputeStages::new(device, compute_pipeline_layout, &shaders);
        let graphics_pipeline =
            create_graphics_pipeline(device, &shaders, graphics_pipeline_layout, render_pass);

        let (framebuffers, allocated_images) = create_framebuffers(
            device,
            memory_allocator,
            compute_command_pool,
            compute_queue,
            extent,
            destination_views,
            render_pass,
        );

        // Create the compute fence to ensure that compute resources can be accessed by the CPU after queue submission.
        let compute_fence = unsafe {
            device.create_fence(
                &ash::vk::FenceCreateInfo::default().flags(ash::vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .expect("Unable to create the compute fence for the fluid simulation");

        // Allocate a new command buffer for the compute operations.
        let buffer_info = ash::vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_command_pool)
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let &compute_command_buffer = unsafe {
            device.allocate_command_buffers(&buffer_info)
        }
                .expect("Unable to allocate the command buffer for the fluid simulation")
                .first()
                .expect("vkAllocateCommandBuffers returned an empty list of command buffers but provided a successful result");

        let (descriptor_pool, descriptor_sets) = allocate_descriptor_sets(
            device,
            &[
                compute_descriptor_set_layout,
                graphics_descriptor_set_layout,
            ],
        );

        update_descriptor_sets(device, &descriptor_sets, &allocated_images);

        Self {
            shaders,
            compute_descriptor_set_layout,
            graphics_descriptor_set_layout,
            compute_pipeline_layout,
            graphics_pipeline_layout,
            render_pass,
            compute_pipelines,
            graphics_pipeline,
            framebuffers,
            allocated_images: allocated_images.into(),
            compute_fence,
            graphics_fence: None,
            compute_command_buffer,
            descriptor_pool,
            descriptor_sets,
            current_display_texture: FluidDisplayTexture::default(),
        }
    }

    /// Destroy the fluid simulation resources.
    pub fn destroy(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);

            device
                .wait_for_fences(&[self.compute_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Unable to wait for the compute fence to signal");
            device.destroy_fence(self.compute_fence, None);

            for image in self.allocated_images.drain(..) {
                image.destroy(device, memory_allocator);
            }
            for framebuffer in &self.framebuffers {
                device.destroy_framebuffer(*framebuffer, None);
            }

            device.destroy_pipeline(self.graphics_pipeline, None);
            self.compute_pipelines.destroy(device);

            device.destroy_render_pass(self.render_pass, None);

            device.destroy_pipeline_layout(self.compute_pipeline_layout, None);
            device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);

            device.destroy_descriptor_set_layout(self.compute_descriptor_set_layout, None);
            device.destroy_descriptor_set_layout(self.graphics_descriptor_set_layout, None);
        }
        self.shaders.destroy(device);
    }

    /// Update device addresses by switching input and output (alpha/beta) buffers.
    fn swap_descriptor_sets(&mut self, device: &ash::Device) {
        self.allocated_images.swap(0, 5);
        self.allocated_images.swap(3, 4);
        self.allocated_images.swap(6, 7);

        let image_infos = [
            ash::vk::DescriptorImageInfo::default()
                .image_view(self.allocated_images[0].image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL),
            ash::vk::DescriptorImageInfo::default()
                .image_view(self.allocated_images[3].image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL),
            ash::vk::DescriptorImageInfo::default()
                .image_view(self.allocated_images[4].image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL),
            ash::vk::DescriptorImageInfo::default()
                .image_view(self.allocated_images[5].image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL),
            ash::vk::DescriptorImageInfo::default()
                .image_view(self.allocated_images[6].image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL),
            ash::vk::DescriptorImageInfo::default()
                .image_view(self.allocated_images[7].image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL),
        ];

        let writes = [
            // Compute descriptor set.
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(0)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[0])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(3)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[1])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(4)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[2])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(5)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[3])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(6)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[4])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(7)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[5])),
            // Graphics descriptor set.
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[1])
                .dst_binding(0)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[3])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[1])
                .dst_binding(1)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[5])),
            ash::vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[1])
                .dst_binding(2)
                .descriptor_count(1)
                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[1])),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Create the framebuffers, likely after a swapchain recreation.
    /// # Safety
    /// * The existing render pass must be valid.
    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
        compute_command_pool: ash::vk::CommandPool,
        compute_queue: ash::vk::Queue,
        extent: ash::vk::Extent2D,
        destination_views: &[ash::vk::ImageView],
    ) {
        // Wait for the compute fence to ensure all resources are available.
        unsafe {
            let fences = [
                self.compute_fence,
                self.graphics_fence.take().unwrap_or_default(),
            ];
            device
                .wait_for_fences(
                    if self.graphics_fence.is_some() {
                        &fences
                    } else {
                        &fences[..1]
                    },
                    true,
                    FIVE_SECONDS_IN_NANOSECONDS,
                )
                .expect(
                    "Failed to wait for the compute or graphics fence for the fluid simulation",
                );
        }

        unsafe {
            device.destroy_fence(self.compute_fence, None);
            self.compute_fence = device
                .create_fence(
                    &ash::vk::FenceCreateInfo::default().flags(ash::vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .expect("Unable to create the compute fence for the fluid simulation");
        }

        for framebuffer in &self.framebuffers {
            unsafe {
                device.destroy_framebuffer(*framebuffer, None);
            }
        }
        for allocated_image in self.allocated_images.drain(..) {
            allocated_image.destroy(device, memory_allocator);
        }
        let (framebuffers, allocated_images) = create_framebuffers(
            device,
            memory_allocator,
            compute_command_pool,
            compute_queue,
            extent,
            destination_views,
            self.render_pass,
        );

        update_descriptor_sets(device, &self.descriptor_sets, &allocated_images);

        self.framebuffers = framebuffers;
        self.allocated_images = allocated_images.into();
    }

    /// Helper to record the compute commands for the fluid simulation to the desired command buffer.
    /// # Safety
    /// The command buffer must not be in the recording state.
    fn create_compute_command_buffer(
        &mut self,
        device: &ash::Device,
        extent: ash::vk::Extent2D,
        push_constants: &ComputePushConstants,
    ) {
        // Ensure that the command buffer is in the recording state.
        unsafe {
            let command_buffer_begin_info = ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.compute_command_buffer, &command_buffer_begin_info)
                .expect(
                    "Failed to begin recording a compute command buffer for the fluid simulation",
                );
        };

        // Helper lambda to add memory barriers to the command buffer.
        let add_barrier = |src_stage_mask: ash::vk::PipelineStageFlags2,
                           src_access_mask: ash::vk::AccessFlags2,
                           dst_stage_mask: ash::vk::PipelineStageFlags2,
                           dst_access_mask: ash::vk::AccessFlags2| {
            let memory_barrier = ash::vk::MemoryBarrier2KHR::default()
                .src_stage_mask(src_stage_mask)
                .src_access_mask(src_access_mask)
                .dst_stage_mask(dst_stage_mask)
                .dst_access_mask(dst_access_mask);
            unsafe {
                device.cmd_pipeline_barrier2(
                    self.compute_command_buffer,
                    &ash::vk::DependencyInfo::default().memory_barriers(&[memory_barrier]),
                );
            }
        };

        // Record the compute commands for each stage of the fluid simulation.
        unsafe {
            device.cmd_push_constants(
                self.compute_command_buffer,
                self.compute_pipeline_layout,
                ash::vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push_constants),
            );

            // Bind the compute descriptor sets.
            device.cmd_bind_descriptor_sets(
                self.compute_command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_layout,
                0,
                &[self.descriptor_sets[0]],
                &[],
            );

            // NOTE: The use of `8` here is directly related to the local group size in the compute shaders.
            // 8*8=64 is a multiple of 64 to accommodate NVIDIA and AMD physical hardware.
            let workgroups_x = extent.width / 8 + u32::from(extent.width % 8 != 0);
            let workgroups_y = extent.height / 8 + u32::from(extent.height % 8 != 0);

            // Apply the curl compute shader.
            device.cmd_bind_pipeline(
                self.compute_command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.curl,
            );
            device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);

            // The vorticity compute shader requires sampling the curl texture, so add a barrier.
            add_barrier(
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_WRITE,
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_READ,
            );

            // Apply the vorticity compute shader.
            device.cmd_bind_pipeline(
                self.compute_command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.vorticity,
            );
            device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);

            // The divergence compute shader requires sampling the velocity texture, so add a barrier.
            add_barrier(
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_WRITE,
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_READ,
            );

            // Apply the divergence compute shader.
            device.cmd_bind_pipeline(
                self.compute_command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.divergence,
            );
            device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);

            // Apply the pressure compute shaders in an iterative loop.
            for _ in 0..MAX_PRESSURE_SMOOTHING_ITERATIONS {
                // Both pressure stages are dependent on the previous pressure stage, so add a barrier.
                add_barrier(
                    ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                    ash::vk::AccessFlags2::SHADER_WRITE,
                    ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                    ash::vk::AccessFlags2::SHADER_READ,
                );

                device.cmd_bind_pipeline(
                    self.compute_command_buffer,
                    ash::vk::PipelineBindPoint::COMPUTE,
                    self.compute_pipelines.alpha_pressure,
                );
                device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);

                // A second-stage barrier.
                add_barrier(
                    ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                    ash::vk::AccessFlags2::SHADER_WRITE,
                    ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                    ash::vk::AccessFlags2::SHADER_READ,
                );

                device.cmd_bind_pipeline(
                    self.compute_command_buffer,
                    ash::vk::PipelineBindPoint::COMPUTE,
                    self.compute_pipelines.beta_pressure,
                );
                device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);
            }

            // The gradient subtract compute shader requires sampling the pressure textures, so add a barrier.
            add_barrier(
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_WRITE,
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_READ,
            );

            // Apply the gradient subtract compute shader.
            device.cmd_bind_pipeline(
                self.compute_command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.gradient_subtract,
            );
            device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);

            // The advection compute shader requires sampling the output velocity texture, so add a barrier.
            add_barrier(
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_WRITE,
                ash::vk::PipelineStageFlags2::COMPUTE_SHADER,
                ash::vk::AccessFlags2::SHADER_READ,
            );

            // Apply the advection compute shader.
            device.cmd_bind_pipeline(
                self.compute_command_buffer,
                ash::vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.advection,
            );
            device.cmd_dispatch(self.compute_command_buffer, workgroups_x, workgroups_y, 1);

            // End the command buffer recording.
            device
                .end_command_buffer(self.compute_command_buffer)
                .expect(
                    "Failed to end recording a compute command buffer for the fluid simulation",
                );
        }
    }

    /// Update the internal active display texture to the next in the cycle.
    pub fn next_display_texture(&mut self) {
        self.current_display_texture = self.current_display_texture.next();

        println!(
            "Switched to the next display texture: {:?}",
            self.current_display_texture
        );
    }

    /// Render the fluid simulation.
    /// # Safety
    /// The `graphics_command_buffer` must be in the recording state to be submitted by the caller.
    pub fn render_frame(
        &mut self,
        device: &ash::Device,
        compute_semaphore: Option<ash::vk::Semaphore>,
        compute_queue: ash::vk::Queue,
        graphics_command_buffer: ash::vk::CommandBuffer,
        extent: ash::vk::Extent2D,
        image_index: usize,
        push_constants: &ComputePushConstants,
        current_graphics_fence: ash::vk::Fence,
    ) {
        // Wait for the compute fence to ensure all resources are available.
        unsafe {
            let fences = [self.compute_fence, self.graphics_fence.unwrap_or_default()];
            device
                .wait_for_fences(
                    if self.graphics_fence.is_some() {
                        &fences
                    } else {
                        &fences[..1]
                    },
                    true,
                    FIVE_SECONDS_IN_NANOSECONDS,
                )
                .expect(
                    "Failed to wait for the compute or graphics fence for the fluid simulation",
                );
            self.graphics_fence = Some(current_graphics_fence);
        }

        // Swap the descriptor set input and output images.
        self.swap_descriptor_sets(device);

        // Record the compute commands for the fluid simulation to the desired command buffer.
        self.create_compute_command_buffer(device, extent, push_constants);

        unsafe {
            // Requires some hoops to satisfy the borrow checker, but sets the command buffer to the submit info.
            let compute_command_buffer = [self.compute_command_buffer];
            let mut semaphores = [ash::vk::Semaphore::null()];
            let mut submit_info =
                ash::vk::SubmitInfo::default().command_buffers(&compute_command_buffer);
            // Optionally, add a semaphore in the case when the compute and graphics queue families are different.
            if let Some(compute_semaphore) = compute_semaphore {
                semaphores[0] = compute_semaphore;
                submit_info = submit_info.signal_semaphores(&semaphores);
            }

            // Reset the compute fence and submit the compute command buffer.
            device
                .reset_fences(&[self.compute_fence])
                .expect("Failed to reset the compute fence for the fluid simulation");
            device
                .queue_submit(compute_queue, &[submit_info], self.compute_fence)
                .expect("Failed to submit the compute command buffer for the fluid simulation");
        }

        // Ensure that the graphics command buffer has the proper push constants and descriptor sets bound.
        unsafe {
            let push_constants = FragmentPushConstants {
                screen_size: [extent.width, extent.height],
                display_texture: self.current_display_texture,
                ..Default::default()
            };
            device.cmd_push_constants(
                graphics_command_buffer,
                self.graphics_pipeline_layout,
                ash::vk::ShaderStageFlags::FRAGMENT,
                0,
                utils::data_byte_slice(&push_constants),
            );

            // Bind the compute descriptor sets.
            device.cmd_bind_descriptor_sets(
                graphics_command_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline_layout,
                0,
                &[self.descriptor_sets[1]],
                &[],
            );
        }

        // Record the graphics commands for the fluid simulation.
        unsafe {
            device.cmd_begin_render_pass(
                graphics_command_buffer,
                &ash::vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[image_index])
                    .render_area(ash::vk::Rect2D {
                        offset: ash::vk::Offset2D { x: 0, y: 0 },
                        extent,
                    }),
                ash::vk::SubpassContents::INLINE,
            );
        }

        // Set the viewport and scissor in the command buffer because we specified they would be set dynamically in the pipeline.
        unsafe {
            device.cmd_set_viewport(
                graphics_command_buffer,
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
                graphics_command_buffer,
                0,
                &[ash::vk::Rect2D {
                    offset: ash::vk::Offset2D::default(),
                    extent,
                }],
            );
        }

        // Bind the graphics pipeline for the fluid simulation.
        unsafe {
            device.cmd_bind_pipeline(
                graphics_command_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
        }

        // Draw the full screen quad.
        unsafe { device.cmd_draw(graphics_command_buffer, 4, 1, 0, 0) };

        // End the render pass.
        unsafe { device.cmd_end_render_pass(graphics_command_buffer) };
    }

    /// Helper for creating new push constants with the given information and buffer addresses.
    pub fn new_push_constants(
        &mut self,
        extent: ash::vk::Extent2D,
        cursor_position: [f32; 2],
        cursor_velocity: [f32; 2],
        cursor_dye: [f32; 4],
        delta_time: f32,
    ) -> ComputePushConstants {
        // Create a new set of push constants for the fluid simulation.
        ComputePushConstants {
            cursor_dye,
            cursor_position,
            cursor_velocity,
            screen_size: [extent.width, extent.height],

            delta_time,
            velocity_diffusion_rate: 0.12,
            dye_diffusion_rate: 1.2,
            vorticity_strength: 22.,
        }
    }
}
