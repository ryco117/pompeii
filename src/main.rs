use std::ffi::CStr;

use winit::{
    event_loop::EventLoop,
    raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _},
};

const WINDOW_TITLE: &str = "Pompeii";
const DEFAULT_WINDOWS_SIZE: winit::dpi::LogicalSize<f32> = winit::dpi::LogicalSize::new(800., 600.);

mod cli;
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
    // Parse the command-line arguments.
    use clap::Parser as _;
    let cli_args = cli::Args::parse();

    // Initialize `event_loop`, the manager of windowing and related events.
    let event_loop = winit::event_loop::EventLoop::<PompeiiEvent>::with_user_event()
        .build()
        .expect("Unable to initialize winit event loop");

    let mut app = PompeiiApp::new(cli_args, &event_loop);
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
            .with_inner_size(size);
    event_loop.create_window(window_attributes)
}

/// App-specific events that can be created and handled.
enum PompeiiEvent {}

/// The state of the application.
enum PompeiiState {
    Empty,
    Windowed {
        window: winit::window::Window,
        renderer: engine::Renderer,
    },
}

/// The main application state and event handler.
struct PompeiiApp {
    args: cli::Args,
    vulkan: utils::VulkanCore,
    state: PompeiiState,
    tick_count: u64,
}

impl PompeiiApp {
    /// Create a new Pompeii application with the given Vulkan API and instance.
    /// Creation of the swapchain and other objects are deferred until the application is resumed, when a window will be available.
    fn new(args: cli::Args, event_loop: &EventLoop<PompeiiEvent>) -> Self {
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

        // Add the ability to check and specify additional device features. This is required for ray-tracing through the `vkGetPhysicalDeviceFeatures2KHR` call.
        extension_names.push(ash::khr::get_physical_device_properties2::NAME);

        // Add the debug utility extension if in debug mode.
        #[cfg(debug_assertions)]
        extension_names.push(ash::ext::debug_utils::NAME);

        // Create a Vulkan instance for our application initialized with the `Empty` state.
        PompeiiApp {
            args,
            vulkan: utils::VulkanCore::new(&extension_names),
            state: PompeiiState::Empty,
            tick_count: 0,
        }
    }

    /// Redraw the window surface if we have initialized the relevant components.
    fn redraw(&mut self) {
        let PompeiiState::Windowed { window, renderer } = &mut self.state else {
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
        // TODO: Consider catching window resizes and minimizes as events, then checking for a flag here.
        //       This would reduce CPU overhead in the draw loop.
        {
            let window_size = window.inner_size();
            let swapchain_size = renderer.swapchain.extent();
            if window_size.width != swapchain_size.width
                || window_size.height != swapchain_size.height
            {
                if window_size.width == 0 || window_size.height == 0 {
                    // Skip all operations if the window contains no pixels.
                    #[cfg(debug_assertions)]
                    println!("Window size is zero, skipping frame");
                } else {
                    #[cfg(debug_assertions)]
                    println!(
                        "Swapchain is out of date at window-size check, needs to be recreated."
                    );

                    renderer.swapchain.recreate_swapchain(
                        &self.vulkan,
                        renderer.physical_device,
                        &renderer.logical_device,
                        renderer.surface,
                        &mut renderer.framebuffers,
                        renderer.swapchain_preferences,
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

        // Attempt to render the frame, or bail and recreate the swapchain if there is a recoverable error.
        renderer.render_frame(&self.vulkan)
    }

    /// Handle keyboard input events.
    fn handle_keyboard_input(&mut self, key_event: winit::event::KeyEvent) {
        let PompeiiState::Windowed { window, .. } = &mut self.state else {
            return;
        };

        // Handle keyboard input events.
        let winit::event::KeyEvent {
            state,
            logical_key: key,
            ..
        } = key_event;
        if state == winit::event::ElementState::Pressed {
            match key.as_ref() {
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                    if matches!(window.fullscreen(), Some(_)) {
                        // Exit fullscreen mode.
                        window.set_fullscreen(None);
                    }
                }
                winit::keyboard::Key::Character("f")
                | winit::keyboard::Key::Named(winit::keyboard::NamedKey::F11) => {
                    if matches!(window.fullscreen(), Some(_)) {
                        // Exit fullscreen mode.
                        window.set_fullscreen(None);
                    } else {
                        // Enter fullscreen mode in borderless mode, defaulting to the active monitor.
                        window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                    }
                }
                _ => (),
            }
        }
    }
}

/// Create a `winit` compliant interface for managing the application state.
impl winit::application::ApplicationHandler<PompeiiEvent> for PompeiiApp {
    /// Create a new windowing system if the application is initialized.
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        println!("Application resuming...");
        if matches!(self.state, PompeiiState::Empty) {
            // Required device extensions for the swapchain.
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
            // TODO: Allow the CLI to specify the image format and color-space preferences.
            let swapchain_preferences = utils::SwapchainPreferences {
                present_mode: Some(self.args.present_mode.into()),
                ..Default::default()
            };
            let swapchain = utils::Swapchain::new(
                &self.vulkan,
                physical_device,
                &logical_device,
                surface,
                swapchain_preferences,
                None,
            );
            let image_count = swapchain.image_count();

            // Create the render pass that will orchestrate usage of the attachments in a framebuffer's render.
            let render_pass = engine::create_render_pass(&logical_device, &swapchain);

            // Process the shaders that will be used in the graphics pipeline.
            let vertex_module = utils::create_shader_module(&logical_device, shaders::VERTEX);
            let fragment_module = utils::create_shader_module(&logical_device, shaders::FRAGMENT);

            // Create the graphics pipeline that will be used to render the application.
            let graphics_pipeline = engine::create_graphics_pipeline(
                &logical_device,
                vertex_module,
                fragment_module,
                render_pass,
            );

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
                renderer: engine::Renderer {
                    physical_device,
                    logical_device,
                    surface,
                    swapchain,
                    render_pass,
                    graphics_pipeline,
                    graphics_queue,
                    presentation_queue,
                    command_buffers,
                    framebuffers,
                    fences_and_state,
                    swapchain_preferences,
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

            // Handle keyboard input events.
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event)
            }

            // Ignore other events.
            _ => (),
        }
    }
}
