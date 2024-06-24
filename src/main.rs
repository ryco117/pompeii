use std::ffi::CStr;

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
use engine::utils::{self, SwapchainPreferences};

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
    user_toggle: bool,
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

        // Attempt to initialize the core Vulkan objects. In case of failure, safely close after the user has seen the error.
        let vulkan = match utils::VulkanCore::new(&extension_names, ash::vk::API_VERSION_1_2) {
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
            user_toggle: false,
        }
    }

    /// Redraw the window surface if we have initialized the relevant components.
    fn redraw(&mut self) {
        let Some(PompeiiGraphics {
            window, renderer, ..
        }) = &mut self.graphics
        else {
            return;
        };

        // Request a redraw of the window surface whenever possible.
        window.request_redraw();

        // Increment the tick count for the application.
        if self.tick_count % TICK_SAMPLING_LENGTH == 0 {
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

                    renderer.recreate_swapchain(
                        &self.vulkan,
                        SwapchainPreferences {
                            preferred_extent: Some(ash::vk::Extent2D {
                                width: window_size.width,
                                height: window_size.height,
                            }),
                            ..renderer.swapchain_preferences
                        },
                    );
                }
                return;
            }
        }

        // Get updated state for drawing.
        // TODO: Update the game state outside of the render loop.
        let push_constants = engine::PushConstants {
            time: self.start_time.elapsed().as_secs_f32(),
        };

        // Attempt to render the frame, or bail and recreate the swapchain if there is a recoverable error.
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

                // Handle the `T` key to toggle the user toggle.
                winit::keyboard::Key::Character("t") => {
                    self.user_toggle = !self.user_toggle;
                    println!("User toggle is now {}", self.user_toggle);

                    // Update the specialization constants for the renderer.
                    if let Some(PompeiiGraphics { renderer, .. }) = &mut self.graphics {
                        renderer.update_specialization_constants(engine::SpecializationConstants {
                            toggle: u32::from(self.user_toggle),
                        });
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
        println!("Window size at creation: {current_extent:?}");

        // Create a renderer specific to this application's needs.
        // NOTE: Some platforms require us to specify the preferred extent for the swapchain.
        let preferred_extent = Some(ash::vk::Extent2D {
            width: current_extent.width,
            height: current_extent.height,
        });
        // TODO: Allow the CLI to specify the image format and color-space preferences.
        let swapchain_preferences = if self.args.hdr && self.vulkan.enabled_colorspace_ext {
            utils::SwapchainPreferences {
                format: Some(ash::vk::Format::A2B10G10R10_UNORM_PACK32),
                color_space: Some(ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT),
                present_mode: Some(self.args.present_mode.into()),
                color_samples: Some(self.args.msaa.into()),
                preferred_extent,
            }
        } else {
            utils::SwapchainPreferences {
                present_mode: Some(self.args.present_mode.into()),
                preferred_extent,
                color_samples: Some(self.args.msaa.into()),
                ..Default::default()
            }
        };
        let renderer = engine::Renderer::new(
            &self.vulkan,
            surface,
            swapchain_preferences,
            engine::SpecializationConstants {
                toggle: u32::from(self.user_toggle),
            },
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

            // Redraw the window surface when requested.
            winit::event::WindowEvent::RedrawRequested => self.redraw(),

            // Handle keyboard input events.
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
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
