use std::ffi::CStr;

use nalgebra_glm as glm;
use smallvec::SmallVec;
use winit::{
    event_loop::EventLoop,
    raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _},
};

/// The title of the main window.
const WINDOW_TITLE: &str = "Pompeii";

/// The default size of the main window.
const DEFAULT_WINDOWS_SIZE: winit::dpi::LogicalSize<f32> = winit::dpi::LogicalSize::new(800., 600.);

/// The number of ticks to sample before printing a message.
const TICK_SAMPLING_LENGTH: u64 = 6_000;

mod cli;
mod engine;
use engine::utils;

fn main() {
    // Parse the command-line arguments.
    use clap::Parser as _;
    let cli_args = cli::Args::parse();

    // Initialize `event_loop`, the manager of windowing and related events.
    let event_loop = winit::event_loop::EventLoop::<PompeiiEvent>::with_user_event()
        .build()
        .expect("Unable to initialize winit event loop");

    // Create the Pompeii application initialized with the CLI arguments and specifics of the event loop at runtime (i.e., platform-dependent windowing).
    let mut app = PompeiiApp::new(cli_args, &event_loop);

    // Run the application event loop.
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

/// The graphics-specific state. Needs to be initialized after the event loop has begun.
struct PompeiiGraphics {
    window: winit::window::Window,
    renderer: engine::Renderer,
}

/// The main application state and event handler.
struct PompeiiApp {
    args: cli::Args,
    vulkan: utils::VulkanCore,
    graphics: Option<PompeiiGraphics>,
    tick_count: u64,
    start_time: std::time::Instant,
    last_frame_time: Option<std::time::Instant>,
    last_mouse_position: Option<(winit::dpi::PhysicalPosition<f64>, std::time::Instant)>,
    mouse_click: Option<[f32; 2]>,
    mouse_velocity: [f32; 2],
}

impl PompeiiApp {
    /// Create a new Pompeii application with the given Vulkan API and instance.
    /// Creation of the swapchain and other objects are deferred until the application is resumed, when a window will be available.
    fn new(args: cli::Args, event_loop: &EventLoop<PompeiiEvent>) -> Self {
        // Get Vulkan instance extensions required by the windowing system, including `VK_KHR_surface` and platform-specific ones.
        let extension_names = ash_window::enumerate_required_extensions(
            event_loop
                .display_handle()
                .expect("Failed to get a display handle")
                .as_raw(),
        )
        .expect("Unable to enumerate required extensions for the window")
        .iter()
        .map(|&e| unsafe { CStr::from_ptr(e) })
        .collect::<SmallVec<[_; utils::EXPECTED_MAX_ENABLED_INSTANCE_EXTENSIONS]>>();

        #[cfg(debug_assertions)]
        println!("INFO: Enabling `ash_window` required instance extensions: {extension_names:?}");

        // Attempt to initialize the core Vulkan objects. In case of failure, safely close after the user has seen the error.
        let vulkan = match utils::VulkanCore::new(&extension_names, &[]) {
            Ok(v) => v,
            Err(e) => {
                use std::io::Write as _; // For `flush` method.

                // Print the error and prompt the user to accept the failure message.
                match e {
                    utils::VulkanCoreError::Loading(e) => eprintln!("Error initializing Vulkan: {e}"),
                    utils::VulkanCoreError::MissingExtension(e) => eprintln!("Error initializing Vulkan: Can't use this Vulkan instance because it doesn't support extension {e}"),
                    utils::VulkanCoreError::MissingLayer(l) => eprintln!("Error initializing Vulkan: Can't use this Vulkan instance because it doesn't support layer {l}"),
                }
                print!("Press enter to exit... ");
                std::io::stdout().flush().unwrap();

                // Wait for the user to press enter before exiting.
                std::io::stdin().read_line(&mut String::new()).unwrap();
                std::process::exit(-1);
            }
        };

        // Create a Vulkan instance for our application initialized with the `Empty` state.
        PompeiiApp {
            args,
            vulkan,
            graphics: None,
            tick_count: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: None,
            last_mouse_position: None,
            mouse_click: None,
            mouse_velocity: [0., 0.],
        }
    }

    /// Update the game state and return the push constants for the next frame.
    /// # Panics
    /// Panics if the `graphics` field is not initialized.
    fn update_gamestate(&mut self) -> engine::DemoPushConstants {
        assert!(self.graphics.is_some(), "Graphics state not initialized");

        // Get updated state for drawing.
        let now = std::time::Instant::now();
        let time = now.duration_since(self.start_time).as_secs_f32();
        let delta_time = self.last_frame_time.map_or(time, |last_frame| {
            now.duration_since(last_frame).as_secs_f32()
        });
        self.last_frame_time = Some(now);
        let extent = self.graphics.as_ref().unwrap().renderer.swapchain_extent();

        // Update the game state and get the per-frame data in the form of push constants.
        let push_constants = match &mut self.graphics.as_mut().unwrap().renderer.active_demo {
            engine::DemoPipeline::Triangle(_) => {
                engine::DemoPushConstants::Triangle(engine::example_triangle::PushConstants {
                    time,
                })
            }

            engine::DemoPipeline::Fluid(fluid) => {
                let dye_cycle = 12. * time;
                let push_constants = fluid.new_push_constants(
                    extent,
                    self.last_mouse_position.map_or([-1024.; 2], |m| m.0.into()),
                    self.mouse_velocity,
                    [
                        ((dye_cycle - 0.7).sin() + 0.5).max(0.) * (2. / 3.),
                        ((-dye_cycle - 0.3).sin() + 0.2).max(0.) * (5. / 6.),
                        (dye_cycle - 0.1).cos().max(0.),
                        f32::from(self.mouse_click.is_none()),
                    ],
                    delta_time,
                );
                engine::DemoPushConstants::Fluid(push_constants)
            }

            engine::DemoPipeline::RayTracing(_) => {
                engine::DemoPushConstants::RayTracing(engine::example_ray_tracing::PushConstants {
                    view_inverse: glm::Mat4::new_rotation(glm::Vec3::new(0., -0.25 * time, 0.))
                        * glm::Mat4::new_translation(&glm::Vec3::new(0., -1., 10.)),
                    time,
                })
            }
        };

        {
            // Decay the constants which were set between the current and last frame.
            let decay = (-8. * delta_time).exp();
            self.mouse_velocity[0] *= decay;
            self.mouse_velocity[1] *= decay;
        }

        // Return the push constants for this frame for consumption by the renderer.
        push_constants
    }

    /// Redraw the window surface if we have initialized the relevant components.
    fn redraw(&mut self, push_constants: &engine::DemoPushConstants) {
        let Some(PompeiiGraphics { renderer, .. }) = &mut self.graphics else {
            return;
        };

        // Increment the tick count for the application.
        if self.tick_count % TICK_SAMPLING_LENGTH == 0 {
            println!(
                "{:?} Tick count: {}",
                std::time::Instant::now(),
                self.tick_count
            );
        }
        self.tick_count += 1;

        // Attempt to render the frame, or bail if there is a recoverable error.
        renderer.render_frame(&self.vulkan, push_constants);
    }

    /// Handle keyboard input events.
    fn handle_keyboard_input(&mut self, key_event: winit::event::KeyEvent) {
        let Some(PompeiiGraphics { window, .. }) = &mut self.graphics else {
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
                // Handle the escape key to exit fullscreen mode.
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                    if window.fullscreen().is_some() {
                        // Exit fullscreen mode.
                        window.set_fullscreen(None);
                    }
                }

                // Handle the `F` key and `F11` to toggle fullscreen mode.
                winit::keyboard::Key::Character("f")
                | winit::keyboard::Key::Named(winit::keyboard::NamedKey::F11) => {
                    if window.fullscreen().is_some() {
                        // Exit fullscreen mode.
                        window.set_fullscreen(None);
                    } else {
                        // Enter fullscreen mode in borderless mode, defaulting to the active monitor.
                        window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                    }
                }

                // Handle the `SPACE` key to toggle the user toggle.
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                    let Some(PompeiiGraphics { renderer, .. }) = &mut self.graphics else {
                        return;
                    };

                    match &mut renderer.active_demo {
                        engine::DemoPipeline::Triangle(t) => {
                            let toggle = (t.specialization_constants().toggle + 1) % 2;

                            // Update the specialization constants for the renderer.
                            renderer.update_specialization_constants(
                                engine::DemoSpecializationConstants::Triangle(
                                    engine::example_triangle::SpecializationConstants { toggle },
                                ),
                            );
                        }
                        engine::DemoPipeline::Fluid(f) => f.next_display_texture(),
                        engine::DemoPipeline::RayTracing(_) => (),
                    }
                }

                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab) => {
                    let Some(PompeiiGraphics { renderer, .. }) = &mut self.graphics else {
                        return;
                    };

                    let ray_tracing_enabled = renderer.ray_tracing_enabled();
                    let new_demo = match renderer.active_demo {
                        // Triangle -> Fluid.
                        engine::DemoPipeline::Triangle(_) => engine::NewDemo::Fluid,

                        // Fluid -> RayTracing.
                        // Skip ray tracing if it's not supported.
                        engine::DemoPipeline::Fluid(_) if ray_tracing_enabled => {
                            engine::NewDemo::RayTracing
                        }

                        // RayTracing -> Triangle.
                        engine::DemoPipeline::Fluid(_) | engine::DemoPipeline::RayTracing(_) => {
                            engine::NewDemo::Triangle(
                                engine::example_triangle::SpecializationConstants::default(),
                            )
                        }
                    };

                    println!("Switching to new demo: {new_demo:?}");
                    renderer.switch_demo(new_demo);
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
        // Some platforms may have circumstances in which the application is resumed after a pause.
        if self.graphics.is_some() {
            return println!("Application resumed");
        }

        println!("Application starting...");

        // Create a new main window for our application.
        let window = create_window(event_loop, WINDOW_TITLE, DEFAULT_WINDOWS_SIZE)
            .expect("Unable to create window");

        // Get a handle to a Vulkan surface for use with the window.
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

        let current_extent = window.inner_size();
        #[cfg(debug_assertions)]
        println!("INFO: Window size at creation: {current_extent:?}");

        // Create a renderer specific to this application's needs.
        // NOTE: Some platforms require us to specify the preferred extent for the swapchain before
        // a current one will be established.
        let preferred_extent = Some(ash::vk::Extent2D {
            width: current_extent.width,
            height: current_extent.height,
        });

        let mut swapchain_preferences = utils::SwapchainPreferences {
            present_mode: Some(self.args.present_mode.into()),
            preferred_extent,
            color_samples: Some(self.args.msaa.into()),
            ..Default::default()
        };

        if self.args.hdr
            && self
                .vulkan
                .enabled_instance_extension(ash::ext::swapchain_colorspace::NAME)
        {
            // TODO: Let swapchain know what kind of format/colorspace we want (HDR/SDR) and let
            // them choose the specifics.
            swapchain_preferences.format = Some(ash::vk::Format::A2B10G10R10_UNORM_PACK32);
            swapchain_preferences.color_space = Some(ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT);
        }

        let renderer = engine::Renderer::new(
            &self.vulkan,
            surface,
            swapchain_preferences,
            engine::DemoSpecializationConstants::Fluid,
            self.args.fxaa,
        );

        // Complete the state transition to windowed mode.
        self.graphics = Some(PompeiiGraphics { window, renderer });

        self.start_time = std::time::Instant::now();
        println!("Application started at {:?}", self.start_time);
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

            // Handle window resizing events.
            winit::event::WindowEvent::Resized(winit::dpi::PhysicalSize { width, height }) => {
                let Some(PompeiiGraphics { renderer, .. }) = &mut self.graphics else {
                    return;
                };

                // Ensure we do not request a swapchain recreation with an area of zero, or the same .
                if (width == 0 || height == 0)
                    || (renderer.swapchain_extent() == ash::vk::Extent2D { width, height })
                {
                    return;
                }

                renderer.swapchain_recreation_required(Some(ash::vk::Extent2D { width, height }));
            }

            // Redraw the window surface when requested.
            winit::event::WindowEvent::RedrawRequested => {
                let Some(PompeiiGraphics {
                    window, renderer, ..
                }) = &mut self.graphics
                else {
                    return;
                };

                // Request a redraw of the window surface whenever possible.
                window.request_redraw();

                // Process any pending swapchain recreation requests.
                renderer.handle_swapchain_resize(&self.vulkan);

                // Check that the current window size won't affect rendering.
                {
                    let extent = renderer.swapchain_extent();
                    let window_size = window.inner_size();
                    if window_size.width == 0 || window_size.height == 0 {
                        // Skip all operations if the window contains no pixels.
                        #[cfg(debug_assertions)]
                        println!("INFO: Window size is zero, skipping frame");
                        return;
                    }

                    let window_size = ash::vk::Extent2D {
                        width: window_size.width,
                        height: window_size.height,
                    };
                    if window_size != extent {
                        eprintln!(
                            "ERROR: Swapchain is out of date at window-size check, needs to be recreated."
                        );

                        renderer.swapchain_recreation_required(Some(window_size));
                        return;
                    }
                }

                // Update the game state and get the push constants for the next frame.
                let push_constants = self.update_gamestate();

                // Submit to the GPU that the next frame be drawn.
                self.redraw(&push_constants);
            }

            // Handle keyboard input events.
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
            }

            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let now = std::time::Instant::now();
                self.mouse_velocity = match self.last_mouse_position {
                    Some((last_position, last_time)) => {
                        let delta_time = now.duration_since(last_time).as_secs_f64();
                        let delta = (
                            (position.x - last_position.x) / delta_time,
                            (position.y - last_position.y) / delta_time,
                        );
                        [
                            0.6 * (delta.0 as f32) + 0.4 * self.mouse_velocity[0],
                            0.6 * (delta.1 as f32) + 0.4 * self.mouse_velocity[1],
                        ]
                    }
                    None => [0.; 2],
                };

                self.last_mouse_position = Some((position, now));
            }
            winit::event::WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state,
                ..
            } => {
                if matches!(state, winit::event::ElementState::Pressed) {
                    self.mouse_click =
                        Some(self.last_mouse_position.map_or([-1024.; 2], |m| m.0.into()));
                } else {
                    self.mouse_click = None;
                }
            }

            // Ignore other events.
            _ => (),
        }
    }
}

impl Drop for PompeiiApp {
    /// Clean up the Vulkan instance and any associated resources.
    fn drop(&mut self) {
        if let Some(graphics) = self.graphics.take() {
            // Wait for the device to finish before cleaning up.
            unsafe {
                graphics
                    .renderer
                    .logical_device
                    .device_wait_idle()
                    .expect("Unable to wait for device idle");
            }

            // Destroy the main Pompeii renderer and Vulkan resources.
            graphics.renderer.destroy(&self.vulkan);
        }

        // Destroy the Vulkan instance.
        unsafe {
            self.vulkan.instance.destroy_instance(None);
        }

        #[cfg(debug_assertions)]
        println!("Freed all Vulkan resources used during the application's lifetime.");
    }
}
