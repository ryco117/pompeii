use std::ffi::CStr;

use winit::raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};

const WINDOW_TITLE: &str = "Pompeii";
const DEFAULT_WINDOWS_SIZE: winit::dpi::LogicalSize<f32> = winit::dpi::LogicalSize::new(800., 600.);

mod utils;

// Store the SPIR-V representation of the shaders in the binary.
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

    // Attempt to dynamically load the Vulkan API from platform-specific shared libraries.
    let vulkan_api = unsafe { ash::Entry::load().expect("Unable to load Vulkan libraries") };

    // Get Vulkan extensions required by the windowing system, including platform-specific ones.
    let mut extension_names = ash_window::enumerate_required_extensions(
        event_loop
            .display_handle()
            .expect("Failed to get a display handle")
            .as_raw(),
    )
    .expect("Unable to enumerate required extensions for the window")
    .to_vec();

    // Add the ability to check and specify additional device features. This is required for ray-tracing.
    extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());

    // Add the debug utility extension if in debug mode.
    #[cfg(debug_assertions)]
    extension_names.push(ash::ext::debug_utils::NAME.as_ptr());

    // Determine which extensions are available at runtime.
    let available_extensions = unsafe {
        vulkan_api
            .enumerate_instance_extension_properties(None)
            .expect("Unable to enumerate available Vulkan extensions")
    };

    // Check that the required extensions are available.
    for ext in &extension_names {
        let ext = unsafe { CStr::from_ptr(*ext) };
        if !available_extensions.iter().any(|a| {
            a.extension_name_as_c_str()
                .expect("Available extension name is not a valid C string")
                == ext
        }) {
            panic!("Required extension {ext:?} is not available. All extensions: {available_extensions:?}");
        }
    }

    // Enable validation layers when using a debug build.
    #[cfg(debug_assertions)]
    let layer_names = {
        // Enable the main validation layer from the Khronos Group when using a debug build.
        const DEBUG_LAYERS: [*const i8; 1] = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

        // Check which layers are available at runtime.
        let available_layers = unsafe {
            vulkan_api
                .enumerate_instance_layer_properties()
                .expect("Unable to enumerate available Vulkan layers")
        };
        println!("Available layers: {available_layers:?}");

        // Check that all the desired debug layers are available.
        if DEBUG_LAYERS.iter().all(|&d| {
            available_layers.iter().any(|a| {
                a.layer_name_as_c_str()
                    .expect("Available layer name is not a valid C string")
                    == unsafe { CStr::from_ptr(d) }
            })
        }) {
            DEBUG_LAYERS
        } else {
            panic!("Required debug layers are not available");
        }
    };
    // Disable validation layers when using a release build.
    #[cfg(not(debug_assertions))]
    let layer_names = [];

    // Create a Vulkan instance with the given extensions and layers.
    let vulkan_instance = {
        let application_info = ash::vk::ApplicationInfo {
            p_application_name: c"Pompeii".as_ptr().cast(),
            p_engine_name: c"Pompeii".as_ptr().cast(),
            api_version: ash::vk::API_VERSION_1_3,
            engine_version: ash::vk::make_api_version(0, 0, 1, 0),
            ..Default::default()
        };
        let instance_info = {
            ash::vk::InstanceCreateInfo {
                p_application_info: &application_info,
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: layer_names.as_ptr(),
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                ..Default::default()
            }
        };
        unsafe {
            vulkan_api
                .create_instance(&instance_info, None)
                .expect("Unable to create Vulkan instance")
        }
    };

    println!("Vulkan instance created");

    let mut app = PompeiiApp::new(vulkan_api, vulkan_instance);
    event_loop.run_app(&mut app);
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
            .with_enabled_buttons(winit::window::WindowButtons::CLOSE | winit::window::WindowButtons::MINIMIZE)
            .with_resizable(false) // TODO: Support window resizing (swapchain recreation).
            .with_inner_size(size);
    event_loop.create_window(window_attributes)
}

/// App-specific events that can be created and handled.
enum PompeiiEvent {}

enum PompeiiState {
    Empty,
    Windowed {
        window: winit::window::Window,
        surface: ash::vk::SurfaceKHR,
    },
}

/// The main application state and event handler.
struct PompeiiApp {
    vulkan: utils::VulkanCore,
    state: PompeiiState,
}

impl PompeiiApp {
    /// Create a new Pompeii application with the given Vulkan API and instance.
    fn new(vulkan_api: ash::Entry, vulkan_instance: ash::Instance) -> Self {
        PompeiiApp {
            vulkan: utils::VulkanCore {
                api: vulkan_api,
                instance: vulkan_instance,
            },
            state: PompeiiState::Empty,
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
            let physical_device = {
                let devices =
                    utils::get_physical_devices(&self.vulkan.instance, &DEVICE_EXTENSIONS);
                devices
                    .first()
                    .expect("Unable to find a suitable physical device")
                    .0
            };

            // Create a logical device capable of rendering to the surface and performing compute.
            let (device, queue_families) =
                utils::new_device(&self.vulkan, physical_device, surface, &DEVICE_EXTENSIONS);
            let utils::QueueFamilies {
                graphics: Some(graphics),
                compute: Some(compute),
                present: Some(present),
                ..
            } = queue_families
            else {
                panic!("Unable to find suitable queue families");
            };

            let command_pool = {
                let command_pool_info = ash::vk::CommandPoolCreateInfo {
                    queue_family_index: graphics,
                    flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                };
                unsafe {
                    device
                        .create_command_pool(&command_pool_info, None)
                        .expect("Unable to create command pool")
                }
            };

            // Complete the state transition to windowed mode.
            self.state = PompeiiState::Windowed { window, surface };
        }

        println!("Application resumed");
    }

    /// Handle OS events to the windowing system.
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let _ = (event_loop, window_id, event);
    }
}
