use crate::engine::utils::{
    self,
    shaders::{ENTRY_POINT_MAIN, FULLSCREEN_VERTEX},
    FIVE_SECONDS_IN_NANOSECONDS,
};

pub mod shaders {
    /// Standard triangle-example fragment shader.
    pub const FLUID_ADVECTION: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid_advection.comp", comp, glsl);
    pub const FLUID_CURL: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid_curl.comp", comp, glsl);
    pub const FLUID_DIVERGENCE: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid_divergence.comp", comp, glsl);
    pub const FLUID_GRADIENT_SUBTRACT: &[u32] = inline_spirv::include_spirv!(
        "src/shaders/example_fluid_gradient_subtract.comp",
        comp,
        glsl
    );
    pub const FLUID_PRESSURE: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid_pressure.comp", comp, glsl);
    pub const FLUID_VORTICITY: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid_vorticity.comp", comp, glsl);
    pub const FLUID_FRAGMENT: &[u32] =
        inline_spirv::include_spirv!("src/shaders/example_fluid.frag", frag, glsl);
}

/// The constant used to define the number of iterations used to adjust the pressure towards a divergence-free field.
/// This value is half the number of iterations used elsewhere because each iteration has two stages which are interleaved.
const MAX_PRESSURE_SMOOTHING_ITERATIONS: u32 = 16;

/// Define the shared push constants for each compute stage of this minimal fluid simulation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
    // GPU device addresses.
    pub input_velocity_buffer: ash::vk::DeviceAddress,
    pub curl_buffer: ash::vk::DeviceAddress,
    pub divergence_buffer: ash::vk::DeviceAddress,
    pub alpha_pressure_buffer: ash::vk::DeviceAddress,
    pub beta_pressure_buffer: ash::vk::DeviceAddress,
    pub output_velocity_buffer: ash::vk::DeviceAddress,
    pub input_dye_buffer: ash::vk::DeviceAddress,
    pub output_dye_buffer: ash::vk::DeviceAddress,

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
#[derive(Clone, Copy, Debug, Default, strum::EnumCount, strum::FromRepr)]
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
#[derive(Debug, Clone, Copy)]
struct FragmentPushConstants {
    // GPU device addresses.
    pub velocity_buffer: ash::vk::DeviceAddress,
    pub dye_buffer: ash::vk::DeviceAddress,
    pub pressure_buffer: ash::vk::DeviceAddress,

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

/// Helper type for managing the resources for an allocated image.
pub struct AllocatedBuffer {
    pub buffer: ash::vk::Buffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub device_address: ash::vk::DeviceAddress,
}
impl AllocatedBuffer {
    /// Create a new image with the given information.
    pub fn new(
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
        buffer_info: &ash::vk::BufferCreateInfo,
        image_debug_name: &str,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    ) -> Self {
        let (buffer, requirements) = unsafe {
            let buffer = device
                .create_buffer(buffer_info, None)
                .expect("Unable to create the buffer for the fluid simulation");
            let mut requirements = device.get_buffer_memory_requirements(buffer);

            // Ensure that the buffer is aligned to 16 bytes.
            // The shaders are each expecting the storage buffers to be aligned to 16 bytes.
            if requirements.alignment > 16 {
                println!(
                    "INFO: Buffer alignment requirement is greater than 16 bytes: {}",
                    requirements.alignment
                );
            } else {
                requirements.alignment = 16;
            }

            (buffer, requirements)
        };

        let allocation = memory_allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: image_debug_name,
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
            })
            .expect("Unable to allocate the buffer for the fluid simulation");
        let memory = unsafe { allocation.memory() };

        // Optionally, set the memory priority to allow the driver to optimize the memory usage.
        if let Some(device_ext) = pageable_device_local_memory {
            unsafe {
                (device_ext.fp().set_device_memory_priority_ext)(device.handle(), memory, 1.0);
            };
        }

        unsafe { device.bind_buffer_memory(buffer, memory, allocation.offset()) }
            .expect("Unable to bind the buffer memory for the fluid simulation");

        let device_address = unsafe {
            device.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(buffer),
            )
        };
        #[cfg(debug_assertions)]
        assert_ne!(
            device_address, 0,
            "The device address of the buffer is zero"
        );

        Self {
            buffer,
            allocation,
            device_address,
        }
    }

    /// Destroy the image and its view.
    pub fn destroy(
        self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        memory_allocator
            .free(self.allocation)
            .expect("Unable to free the image allocation");
    }
}

/// Create a framebuffer that can be used with this render pass.
pub fn create_framebuffers(
    device: &ash::Device,
    memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    extent: ash::vk::Extent2D,
    destination_views: &[ash::vk::ImageView],
    render_pass: ash::vk::RenderPass,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
) -> (Vec<ash::vk::Framebuffer>, [AllocatedBuffer; 8]) {
    // Create several images for storing the partial results of the fluid simulation each frame.
    // These images are not strictly part of the framebuffers, but are used in the fluid simulation and dependent on the size of the surface.
    let mut buffer_info = ash::vk::BufferCreateInfo::default().usage(
        ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    );
    let pixel_count = u64::from(extent.width) * u64::from(extent.height);

    buffer_info.size = pixel_count * std::mem::size_of::<[f32; 2]>() as u64;
    let input_velocity_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim input velocity buffer",
        pageable_device_local_memory,
    );

    buffer_info.size = pixel_count * std::mem::size_of::<f32>() as u64;
    let curl_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim curl buffer",
        pageable_device_local_memory,
    );
    let divergence_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim divergence buffer",
        pageable_device_local_memory,
    );
    let alpha_pressure_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim alpha pressure buffer",
        pageable_device_local_memory,
    );
    let beta_pressure_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim beta pressure buffer",
        pageable_device_local_memory,
    );

    buffer_info.size = pixel_count * std::mem::size_of::<[f32; 2]>() as u64;
    let output_velocity_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim output velocity buffer",
        pageable_device_local_memory,
    );

    buffer_info.size = pixel_count * std::mem::size_of::<[f32; 4]>() as u64;
    let input_dye_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim input dye buffer",
        pageable_device_local_memory,
    );
    let output_dye_image = AllocatedBuffer::new(
        device,
        memory_allocator,
        &buffer_info,
        "Fluid Sim output dye buffer",
        pageable_device_local_memory,
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

    (
        framebuffers,
        [
            input_velocity_image,
            curl_image,
            divergence_image,
            alpha_pressure_image,
            beta_pressure_image,
            output_velocity_image,
            input_dye_image,
            output_dye_image,
        ],
    )
}

/// Create the compute and graphics pipeline layouts.
fn create_pipeline_layout(device: &ash::Device) -> [ash::vk::PipelineLayout; 2] {
    // Create the push constant specification for the fluid simulation. The same data is used for both the compute and fragment shaders.
    let compute_push_constant_range = ash::vk::PushConstantRange {
        stage_flags: ash::vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: std::mem::size_of::<PushConstants>() as u32,
    };

    // Create the pipeline layout for the fluid simulation.
    let compute_pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &ash::vk::PipelineLayoutCreateInfo::default()
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

/// The fluid simulation renderer and resources.
pub struct FluidSimulation {
    shaders: FluidShaders,
    compute_pipeline_layout: ash::vk::PipelineLayout,
    graphics_pipeline_layout: ash::vk::PipelineLayout,
    render_pass: ash::vk::RenderPass,
    compute_pipelines: FluidComputeStages,
    graphics_pipeline: ash::vk::Pipeline,
    framebuffers: Vec<ash::vk::Framebuffer>,
    allocated_images: Vec<AllocatedBuffer>,
    compute_fence: ash::vk::Fence,
    graphics_fence: Option<ash::vk::Fence>,
    compute_command_buffer: ash::vk::CommandBuffer,
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
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    ) -> Self {
        let shaders = FluidShaders::new(device);
        let [compute_pipeline_layout, graphics_pipeline_layout] = create_pipeline_layout(device);

        let render_pass = create_render_pass(device, image_format, destination_layout);

        let compute_pipelines = FluidComputeStages::new(device, compute_pipeline_layout, &shaders);
        let graphics_pipeline =
            create_graphics_pipeline(device, &shaders, graphics_pipeline_layout, render_pass);

        let (framebuffers, allocated_images) = create_framebuffers(
            device,
            memory_allocator,
            extent,
            destination_views,
            render_pass,
            pageable_device_local_memory,
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

        Self {
            shaders,
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
        }
        self.shaders.destroy(device);
    }

    /// Create the framebuffers, likely after a swapchain recreation.
    /// # Safety
    /// * The existing render pass must be valid.
    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
        extent: ash::vk::Extent2D,
        destination_views: &[ash::vk::ImageView],
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
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
            extent,
            destination_views,
            self.render_pass,
            pageable_device_local_memory,
        );
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
        push_constants: &PushConstants,
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
                    &ash::vk::DependencyInfoKHR::default().memory_barriers(&[memory_barrier]),
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
                utils::data_byte_slice(push_constants),
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
        push_constants: &PushConstants,
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

        // Ensure that the graphics command buffer has the proper push constants bound.
        unsafe {
            let push_constants = FragmentPushConstants {
                velocity_buffer: self.allocated_images[5].device_address,
                dye_buffer: self.allocated_images[7].device_address,
                pressure_buffer: self.allocated_images[3].device_address,
                screen_size: [extent.width, extent.height],
                display_texture: self.current_display_texture,
            };
            device.cmd_push_constants(
                graphics_command_buffer,
                self.graphics_pipeline_layout,
                ash::vk::ShaderStageFlags::FRAGMENT,
                0,
                utils::data_byte_slice(&push_constants),
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

        // Update device addresses by switching input and output (alpha/beta) buffers.
        self.allocated_images.swap(0, 5);
        self.allocated_images.swap(3, 4);
        self.allocated_images.swap(6, 7);
    }

    /// Helper for creating new push constants with the given information and buffer addresses.
    pub fn new_push_constants(
        &mut self,
        extent: ash::vk::Extent2D,
        cursor_position: [f32; 2],
        cursor_velocity: [f32; 2],
        cursor_dye: [f32; 4],
        delta_time: f32,
    ) -> PushConstants {
        // Create a new set of push constants for the fluid simulation.
        PushConstants {
            input_velocity_buffer: self.allocated_images[0].device_address,
            curl_buffer: self.allocated_images[1].device_address,
            divergence_buffer: self.allocated_images[2].device_address,
            alpha_pressure_buffer: self.allocated_images[3].device_address,
            beta_pressure_buffer: self.allocated_images[4].device_address,
            output_velocity_buffer: self.allocated_images[5].device_address,
            input_dye_buffer: self.allocated_images[6].device_address,
            output_dye_buffer: self.allocated_images[7].device_address,

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
