use std::ffi::CStr;

use winit::{
    event_loop::EventLoop,
    raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _},
};

const WINDOW_TITLE: &str = "Pompeii";
const DEFAULT_WINDOWS_SIZE: winit::dpi::LogicalSize<f32> = winit::dpi::LogicalSize::new(800., 600.);

mod engine;
use engine::utils;

/// Store the SPIR-V representation of the shaders in the binary.
mod shaders {
    pub const VERTEX: &'static [u32] =
        inline_spirv::include_spirv!("src/shaders/bb_triangle_vert.glsl", vert, glsl);
    pub const FRAGMENT: &'static [u32] =
        inline_spirv::include_spirv!("src/shaders/bb_triangle_frag.glsl", frag, glsl);
}

fn main() {
    // Initialize `event_loop`, the manager of windowing and related events.
    let event_loop = winit::event_loop::EventLoop::<PompeiiEvent>::with_user_event()
        .build()
        .expect("Unable to initialize winit event loop");

    let mut app = PompeiiApp::new(&event_loop);
    event_loop
        .run_app(&mut app)
        .expect("Error during event loop");
}

/// Create a window with the given title and size.
fn create_window(
    event_loop: &winit::event_loop::ActiveEventLoop,
    title: &str,
    size: winit::dpi::LogicalSize<f32>,
) -> Result<winit::window::Window, winit::error::OsError> {
    let window_attributes: winit::window::WindowAttributes =
        winit::window::Window::default_attributes()
            .with_title(title)
            .with_enabled_buttons(
                winit::window::WindowButtons::CLOSE | winit::window::WindowButtons::MINIMIZE,
            )
            .with_resizable(false) // TODO: Support window resizing (swapchain recreation).
            .with_inner_size(size);
    event_loop.create_window(window_attributes)
}

/// App-specific events that can be created and handled.
enum PompeiiEvent {}

struct PompeiiRenderer {
    physical_device: ash::vk::PhysicalDevice,
    logical_device: ash::Device,
    render_pass: ash::vk::RenderPass,
    graphics_pipeline: ash::vk::Pipeline,
    graphics_queue: ash::vk::Queue,
    presentation_queue: ash::vk::Queue,
    command_buffers: Vec<ash::vk::CommandBuffer>,
    framebuffers: Vec<ash::vk::Framebuffer>,
    fences_and_state: Vec<(ash::vk::Fence, bool)>,
}

/// The state of the application.
enum PompeiiState {
    Empty,
    Windowed {
        window: winit::window::Window,
        surface: ash::vk::SurfaceKHR,
        swapchain: utils::Swapchain,
        renderer: PompeiiRenderer,
    },
}

/// The main application state and event handler.
struct PompeiiApp {
    vulkan: utils::VulkanCore,
    state: PompeiiState,
    tick_count: u64,
}

impl PompeiiApp {
    /// Create a new Pompeii application with the given Vulkan API and instance.
    /// Creation of the swapchain and other objects are deferred until the application is resumed, when a window will be available.
    fn new(event_loop: &EventLoop<PompeiiEvent>) -> Self {
        // Get Vulkan extensions required by the windowing system, including platform-specific ones.
        let mut extension_names = ash_window::enumerate_required_extensions(
            event_loop
                .display_handle()
                .expect("Failed to get a display handle")
                .as_raw(),
        )
        .expect("Unable to enumerate required extensions for the window")
        .iter()
        .map(|&e| unsafe { CStr::from_ptr(e) })
        .collect::<Vec<_>>();

        // Add the ability to check and specify additional device features. This is required for ray-tracing.
        extension_names.push(ash::khr::get_physical_device_properties2::NAME);

        // Add the debug utility extension if in debug mode.
        #[cfg(debug_assertions)]
        extension_names.push(ash::ext::debug_utils::NAME);

        // Create a Vulkan instance for our application initialized with the `Empty` state.
        PompeiiApp {
            vulkan: utils::VulkanCore::new(&extension_names),
            state: PompeiiState::Empty,
            tick_count: 0,
        }
    }

    /// Redraw the window surface if we have initialized the relevant components.
    fn redraw(&mut self) {
        let PompeiiState::Windowed {
            window,
            surface,
            swapchain,
            renderer,
        } = &mut self.state
        else {
            return;
        };

        // Request a redraw of the window surface whenever possible.
        window.request_redraw();

        // Increment the tick count for the application.
        if self.tick_count % 6_000 == 0 {
            println!(
                "{:?} Tick count: {}",
                std::time::Instant::now(),
                self.tick_count
            );
        }
        self.tick_count += 1;

        // Check that the current window size won't affect rendering.
        {
            let window_size = window.inner_size();
            let swapchain_size = swapchain.extent();
            if window_size.width != swapchain_size.width
                || window_size.height != swapchain_size.height
            {
                if window_size.width == 0 || window_size.height == 0 {
                    // Skip all operations if the window contains no pixels.
                    #[cfg(debug_assertions)]
                    println!("Window size is zero, skipping frame");
                } else {
                    println!(
                        "Swapchain is out of date at window-size check, needs to be recreated."
                    );
                    swapchain.recreate_swapchain(
                        &self.vulkan,
                        renderer.physical_device,
                        &renderer.logical_device,
                        *surface,
                        &mut renderer.framebuffers,
                        |img, extent| {
                            engine::create_framebuffer(
                                &renderer.logical_device,
                                renderer.render_pass,
                                img,
                                extent,
                            )
                        },
                    );
                }
                return;
            }
        }

        // Get the next image to render to. Has internal synchronization to ensure the previous acquire completed on the GPU.
        let (_image, image_index) = match swapchain.acquire_next_image(&renderer.logical_device) {
            Ok((image, index, false)) => (image, index),

            // TODO: Consider accepting suboptimal for this draw, but set a flag to recreate the swapchain next frame.
            Ok((_, _, true)) | Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let khr_instance =
                    ash::khr::surface::Instance::new(&self.vulkan.api, &self.vulkan.instance);
                let surface_capabilities = unsafe {
                    khr_instance
                        .get_physical_device_surface_capabilities(
                            renderer.physical_device,
                            *surface,
                        )
                        .expect("Unable to get surface capabilities")
                };
                if surface_capabilities.max_image_extent.width == 0
                    || surface_capabilities.max_image_extent.height == 0
                {
                    #[cfg(debug_assertions)]
                    println!("Surface capabilities are zero at image acquire, skipping swapchain recreation");
                    return;
                }

                println!("Swapchain is out of date at image acquire, needs to be recreated.");
                swapchain.recreate_swapchain(
                    &self.vulkan,
                    renderer.physical_device,
                    &renderer.logical_device,
                    *surface,
                    &mut renderer.framebuffers,
                    |img, extent| {
                        engine::create_framebuffer(
                            &renderer.logical_device,
                            renderer.render_pass,
                            img,
                            extent,
                        )
                    },
                );
                return;
            }

            Err(e) => panic!("Unable to acquire next image from swapchain: {e}"),
        };

        // Synchronize the CPU with the GPU for the resources previously used for this image index.
        // TODO: Make this into its own utils function because it is a common pattern.
        // TODO: Example code [here](https://github.com/PacktPublishing/The-Modern-Vulkan-Cookbook/blob/81d96f600eeb54cbaa0c84967d233ab000841ab5/source/vulkancore/CommandQueueManager.cpp#L130)
        //       uses different indices than the swapchain image to synchronize the fences and the command buffers. Consider whether this would be necessary or beneficial.
        let resource_fence = renderer.fences_and_state[image_index as usize].0;
        let command_buffer = renderer.command_buffers[image_index as usize];
        unsafe {
            renderer
                .logical_device
                .wait_for_fences(&[resource_fence], true, utils::FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Unable to wait for fence to begin frame");
            renderer
                .logical_device
                .reset_command_buffer(
                    command_buffer,
                    ash::vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Unable to reset command buffer");
            renderer
                .logical_device
                .begin_command_buffer(
                    command_buffer,
                    &ash::vk::CommandBufferBeginInfo {
                        flags: ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .expect("Unable to begin command buffer");
        }

        // Begin the render pass for the current frame.
        unsafe {
            renderer.logical_device.cmd_begin_render_pass(
                command_buffer,
                &ash::vk::RenderPassBeginInfo {
                    render_pass: renderer.render_pass,
                    framebuffer: renderer.framebuffers[image_index as usize],
                    render_area: ash::vk::Rect2D {
                        offset: ash::vk::Offset2D::default(),
                        extent: swapchain.extent(),
                    },
                    clear_value_count: 1,
                    p_clear_values: &ash::vk::ClearValue::default(),
                    ..Default::default()
                },
                ash::vk::SubpassContents::INLINE,
            );
        }

        // Bind the graphics pipeline to the command buffer.
        unsafe {
            renderer.logical_device.cmd_bind_pipeline(
                command_buffer,
                ash::vk::PipelineBindPoint::GRAPHICS,
                renderer.graphics_pipeline,
            );
        }

        // TODO: Update descriptor sets here when we have them.

        // Draw the example triangle.
        unsafe {
            renderer.logical_device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        // End the render pass for the current frame.
        unsafe {
            renderer.logical_device.cmd_end_render_pass(command_buffer);
            renderer
                .logical_device
                .end_command_buffer(command_buffer)
                .expect("Unable to end the graphics command buffer");
        }

        // Submit the draw command buffer to the GPU.
        let submit_info = ash::vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &swapchain.image_available(),
            p_wait_dst_stage_mask: &ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 1,
            p_signal_semaphores: &swapchain.image_rendered(),
            ..Default::default()
        };
        unsafe {
            renderer
                .logical_device
                .reset_fences(&[resource_fence])
                .expect("Unable to reset fence for this frame's resources");
            renderer
                .logical_device
                .queue_submit(renderer.graphics_queue, &[submit_info], resource_fence)
                .expect("Unable to submit command buffer");

            // Note that this fence has been submitted.
            // TODO: Determine if this is useful.
            renderer.fences_and_state[image_index as usize].1 = true;
        }

        // Queue the presentation of the swapchain image.
        match swapchain.present(renderer.presentation_queue) {
            Ok(false) => (),
            Ok(true) | Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // TODO: Refactor into a function for all swapchain recreation.
                let khr_instance =
                    ash::khr::surface::Instance::new(&self.vulkan.api, &self.vulkan.instance);
                let surface_capabilities = unsafe {
                    khr_instance
                        .get_physical_device_surface_capabilities(
                            renderer.physical_device,
                            *surface,
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
                swapchain.recreate_swapchain(
                    &self.vulkan,
                    renderer.physical_device,
                    &renderer.logical_device,
                    *surface,
                    &mut renderer.framebuffers,
                    |img, extent| {
                        engine::create_framebuffer(
                            &renderer.logical_device,
                            renderer.render_pass,
                            img,
                            extent,
                        )
                    },
                );
                return;
            }
            Err(e) => panic!("Unable to present swapchain image: {:?}", e),
        }
    }
}

/// Create a `winit` compliant interface for managing the application state.
impl winit::application::ApplicationHandler<PompeiiEvent> for PompeiiApp {
    /// Create a new windowing system if the application is initialized.
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        println!("Application resuming...");
        if matches!(self.state, PompeiiState::Empty) {
            const DEVICE_EXTENSIONS: [*const i8; 1] = [ash::khr::swapchain::NAME.as_ptr()];

            let window = create_window(event_loop, WINDOW_TITLE, DEFAULT_WINDOWS_SIZE)
                .expect("Unable to create window");

            let surface = unsafe {
                ash_window::create_surface(
                    &self.vulkan.api,
                    &self.vulkan.instance,
                    window
                        .display_handle()
                        .expect("Failed to get a display handle")
                        .into(),
                    window
                        .window_handle()
                        .expect("Failed to get a window handle")
                        .into(),
                    None,
                )
                .expect("Unable to create Vulkan surface")
            };

            // Use simple heuristics to find the best suitable physical device.
            let (physical_device, device_properties) = {
                let devices =
                    utils::get_physical_devices(&self.vulkan.instance, &DEVICE_EXTENSIONS);

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
                utils::new_device(&self.vulkan, physical_device, surface, &DEVICE_EXTENSIONS);
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
            // TODO: Allow for a command-line argument to set the present mode.
            let swapchain = utils::Swapchain::new(
                &self.vulkan,
                physical_device,
                &logical_device,
                surface,
                None,
                None,
                Some(ash::vk::PresentModeKHR::MAILBOX),
                None,
            );
            let image_count = swapchain.image_count();
            #[cfg(debug_assertions)]
            {
                let present_mode = swapchain.present_mode();
                println!("Present mode: {present_mode:?} * {image_count}\n")
            }

            // Create the render pass that will orchestrate usage of the attachments in a framebuffer's render.
            let render_pass = engine::create_render_pass(&logical_device, &swapchain);

            // Process the shaders that will be used in the graphics pipeline.
            let vertex_module = utils::create_shader_module(&logical_device, shaders::VERTEX);
            let fragment_module = utils::create_shader_module(&logical_device, shaders::FRAGMENT);

            // Create the graphics pipeline that will be used to render the application.
            let graphics_pipeline = engine::create_graphics_pipeline(
                &swapchain,
                &logical_device,
                vertex_module,
                fragment_module,
                render_pass,
            );

            // Create a pool for allocating new commands.
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
            // TODO: Create framebuffers with the render pass because the render pass defines the framebuffer topology.
            let extent = swapchain.extent();
            let framebuffers = swapchain
                .image_views()
                .iter()
                .map(|image_view| {
                    engine::create_framebuffer(&logical_device, render_pass, image_view, extent)
                })
                .collect();

            // Complete the state transition to windowed mode.
            self.state = PompeiiState::Windowed {
                window,
                surface,
                swapchain,
                renderer: PompeiiRenderer {
                    physical_device,
                    logical_device,
                    render_pass,
                    graphics_pipeline,
                    graphics_queue,
                    presentation_queue,
                    command_buffers,
                    framebuffers,
                    fences_and_state,
                },
            };
        }

        println!("Application resumed");
    }

    /// Handle OS events to the windowing system.
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            // Handle application close requests.
            winit::event::WindowEvent::CloseRequested => {
                println!("Window close requested");
                event_loop.exit();
            }

            // Redraw the window surface when requested.
            winit::event::WindowEvent::RedrawRequested => self.redraw(),

            // Ignore other events.
            _ => (),
        }
    }
}
