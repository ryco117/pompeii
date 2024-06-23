use std::{ffi::CStr, num::NonZeroU32};

use smallvec::SmallVec;
use strum::EnumCount as _;

/// Set a sane value for the maximum expected number of queue families.
/// A heap allocation is required if the number of queue families exceeds this value.
const EXPECTED_MAX_QUEUE_FAMILIES: usize = 8;

/// Set a sane value for the maximum expected number of instance extensions.
/// A heap allocation is required if the number of instance extensions exceeds this value.
const EXPECTED_MAX_INSTANCE_EXTENSIONS: usize = 16;

/// Set a sane value for the maximum expected number of physical devices.
/// A heap allocation is required if the number of physical devices exceeds this value.
const EXPECTED_MAX_VULKAN_PHYSICAL_DEVICES: usize = 4;

/// The number of nanoseconds in five seconds.
pub const FIVE_SECONDS_IN_NANOSECONDS: u64 = 5_000_000_000;

/// The main Vulkan library interface. Contains the entry to the Vulkan library and an instance for this app.
pub struct VulkanCore {
    pub api: ash::Entry,
    pub instance: ash::Instance,
    pub khr: ash::khr::surface::Instance,
    pub enabled_colorspace_ext: bool,
}

impl VulkanCore {
    /// Create a new `VulkanCore` with the specified extensions.
    pub fn new(extension_names: &[&CStr]) -> Self {
        // Attempt to dynamically load the Vulkan API from platform-specific shared libraries.
        let vulkan_api = unsafe { ash::Entry::load().expect("Unable to load Vulkan libraries") };

        // Determine which extensions are available at runtime.
        let available_extensions = unsafe {
            vulkan_api
                .enumerate_instance_extension_properties(None)
                .expect("Unable to enumerate available Vulkan extensions")
        };

        #[cfg(debug_assertions)]
        println!("Available instance extensions: {available_extensions:?}\n");

        let available_contains = |ext: &CStr| {
            available_extensions.iter().any(|p| {
                p.extension_name_as_c_str()
                    .expect("Available extension name is not a valid C string")
                    == ext
            })
        };

        // Check that the required extensions are available.
        for &ext in extension_names {
            assert!(available_contains(ext), "Required extension {ext:?} is not available. All extensions: {available_extensions:?}");
        }

        let mut extension_name_pointers: SmallVec<[*const i8; EXPECTED_MAX_INSTANCE_EXTENSIONS]> =
            extension_names.iter().map(|&e| e.as_ptr()).collect();

        // Optionally, enable `VK_EXT_swapchain_colorspace` if is is available and the dependent `VK_KHR_surface` is requested.
        let supports_colorspace_ext = extension_names.contains(&ash::khr::surface::NAME)
            && available_contains(ash::ext::swapchain_colorspace::NAME);
        if supports_colorspace_ext {
            #[cfg(debug_assertions)]
            println!("Enabling VK_EXT_swapchain_colorspace extension");

            extension_name_pointers.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
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
            println!("Available layers: {available_layers:?}\n");

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
                    enabled_extension_count: extension_name_pointers.len() as u32,
                    pp_enabled_extension_names: extension_name_pointers.as_ptr(),
                    ..Default::default()
                }
            };
            unsafe {
                vulkan_api
                    .create_instance(&instance_info, None)
                    .expect("Unable to create Vulkan instance")
            }
        };

        let khr = ash::khr::surface::Instance::new(&vulkan_api, &vulkan_instance);
        Self {
            api: vulkan_api,
            instance: vulkan_instance,
            khr,
            enabled_colorspace_ext: supports_colorspace_ext,
        }
    }
}

/// Query the extended properties of a physical device to determine if it supports ray tracing.
//  TODO: Consider refactoring to allow chaining of device features before performing the final query.
pub fn physical_supports_rtx(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
) -> Option<ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR> {
    // Query the physical device for ray tracing support features.
    let mut ray_query = ash::vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut ray_tracing = ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
        p_next: std::ptr::from_mut(&mut ray_query).cast(),
        ..Default::default()
    };
    let mut acceleration_structure = ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
        p_next: std::ptr::from_mut(&mut ray_tracing).cast(),
        ..Default::default()
    };
    let mut features = ash::vk::PhysicalDeviceFeatures2 {
        p_next: std::ptr::from_mut(&mut acceleration_structure).cast(),
        ..Default::default()
    };
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };

    // If the physical device supports ray tracing, return the ray tracing pipeline properties.
    if acceleration_structure.acceleration_structure > 0
        && ray_tracing.ray_tracing_pipeline > 0
        && ray_query.ray_query > 0
    {
        let mut ray_tracing_pipeline =
            ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut properties = ash::vk::PhysicalDeviceProperties2 {
            p_next: std::ptr::from_mut(&mut ray_tracing_pipeline).cast(),
            ..Default::default()
        };
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
        Some(ray_tracing_pipeline)
    } else {
        None
    }
}

/// Get all physical devices that support Vulkan 1.3 and the required extensions. Sort them by their likelihood of being the desired device.
pub fn get_sorted_physical_devices(
    instance: &ash::Instance,
    required_extensions: &[*const i8],
) -> SmallVec<
    [(ash::vk::PhysicalDevice, ash::vk::PhysicalDeviceProperties);
        EXPECTED_MAX_VULKAN_PHYSICAL_DEVICES],
> {
    /// A helper for scoring a device's dedication to graphics processing.
    fn score_device_type(device_type: ash::vk::PhysicalDeviceType) -> u8 {
        match device_type {
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            ash::vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            ash::vk::PhysicalDeviceType::CPU => 1,
            _ => 0,
        }
    }

    // Get all physical devices that support Vulkan 1.3.
    // TODO: Investigate if `enumerate_physical_devices` can be replaced with something I can pass a `SmallVec` ref to.
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Unable to enumerate physical devices")
    };

    let mut physical_devices = physical_devices
        .into_iter()
        .filter_map(|device| {
            // Query basic properties of the physical device.
            let properties = unsafe { instance.get_physical_device_properties(device) };

            // Query for the basic extensions available to this physical device.
            // Note that instance extension `VK_KHR_get_physical_device_properties2` enables use of function `vkGetPhysicalDeviceFeatures2KHR` for additional features, but is not called here.
            let extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Unable to enumerate device extensions")
            };

            // Ensure every device has their properties printed in debug mode. Called before filtering starts.
            #[cfg(debug_assertions)]
            println!("Physical device {device:?}: {properties:?}\nPhysical device available extensions: {extensions:?}\n");

            // Ensure the device supports Vulkan 1.3.
            if properties.api_version < ash::vk::API_VERSION_1_3 {
                #[cfg(debug_assertions)]
                println!("Physical device {device:?} does not support a sufficiently high Vulkan version");
                return None;
            }

            // Ensure the device supports all required extensions.
            if required_extensions.iter().all(|&req| {
                let req = unsafe { CStr::from_ptr(req) };
                let exists = extensions.iter().any(|e| {
                    e.extension_name_as_c_str()
                        .expect("Available extension name is not a valid C string")
                        == req
                });

                #[cfg(debug_assertions)]
                if !exists {
                    println!("Physical device {device:?} does not support required extension '{req:?}'");
                }
                exists
            }) {
                Some((device, properties))
            } else {
                None
            }
        })
        .collect::<SmallVec<[(ash::vk::PhysicalDevice, ash::vk::PhysicalDeviceProperties); 4]>>();

    #[cfg(debug_assertions)]
    if physical_devices.spilled() {
        println!(
            "INFO: Physical devices list has spilled over to the heap. Device count {} greater than inline size {}",
            physical_devices.len(),
            physical_devices.inline_size(),
        );
    }

    // Sort the physical devices by the device type with preference for GPU's, then descending by graphics dedication.
    physical_devices.sort_by(|(_, a), (_, b)| {
        // Sorting order is reversed (`b.cmp(a)`) to sort the highest scoring device first.
        let device_type_cmp =
            score_device_type(b.device_type).cmp(&score_device_type(a.device_type));

        // If device types are equivalent, then sort by the maximum push constants size.
        if device_type_cmp == std::cmp::Ordering::Equal {
            b.limits
                .max_push_constants_size
                .cmp(&a.limits.max_push_constants_size)
        } else {
            device_type_cmp
        }
    });

    physical_devices
}

/// Available queue family types and their indices.
/// Also, a map of queue family indices to the number of queues that may be and are currently allocated.
pub struct QueueFamilies {
    pub graphics: Vec<u32>,
    pub compute: Vec<u32>,
    pub present: Vec<u32>,
    pub queue_families: Vec<ash::vk::QueueFamilyProperties>,
}

#[derive(strum::EnumCount)]
#[repr(usize)]
enum QueueType {
    Graphics,
    Compute,
    Present,
}

/// Get the necessary queue family indices for a logical device capable of graphics, compute, and presentation.
pub fn get_queue_families(
    vulkan: &VulkanCore,
    physical_device: ash::vk::PhysicalDevice,
    surface: ash::vk::SurfaceKHR,
) -> QueueFamilies {
    // Get the list of available queue families for this device.
    let queue_families = unsafe {
        vulkan
            .instance
            .get_physical_device_queue_family_properties(physical_device)
    };

    #[cfg(debug_assertions)]
    println!("Queue families: {queue_families:?}\n");

    // Find the first queue families for each desired queue type.
    // Use an array with indices to allow compile-time guarantees about the number of queue types.
    let mut all_queue_families = [const { Vec::<u32>::new() }; QueueType::COUNT];
    for (family_index, queue_family) in queue_families.iter().enumerate() {
        // Get a present queue family.
        if all_queue_families[QueueType::Present as usize].is_empty() {
            let is_present_supported = unsafe {
                vulkan
                    .khr
                    .get_physical_device_surface_support(
                        physical_device,
                        family_index as u32,
                        surface,
                    )
                    .expect("Unable to check if 'present' is supported")
            };
            if is_present_supported {
                all_queue_families[QueueType::Present as usize].push(family_index as u32);
            }
        }

        // Get a graphics queue family.
        if queue_family
            .queue_flags
            .contains(ash::vk::QueueFlags::GRAPHICS)
        {
            all_queue_families[QueueType::Graphics as usize].push(family_index as u32);
        }

        // Get a compute queue family.
        if queue_family
            .queue_flags
            .contains(ash::vk::QueueFlags::COMPUTE)
        {
            all_queue_families[QueueType::Compute as usize].push(family_index as u32);
        }
    }

    // TODO: Allow the caller to choose which queue types are needed.
    let graphics = std::mem::take(&mut all_queue_families[QueueType::Graphics as usize]);
    let compute = std::mem::take(&mut all_queue_families[QueueType::Compute as usize]);
    let present = std::mem::take(&mut all_queue_families[QueueType::Present as usize]);

    QueueFamilies {
        graphics,
        compute,
        present,
        queue_families,
    }
}

/// Create a Vulkan logical device capable of graphics, compute, and presentation queues.
/// Returns the device and the queue family indices that were requested for use.
pub fn new_device(
    vulkan: &VulkanCore,
    physical_device: ash::vk::PhysicalDevice,
    surface: ash::vk::SurfaceKHR,
    device_extensions: &[*const i8],
) -> (ash::Device, QueueFamilies) {
    // Get the necessary queue family indices for the logical device.
    let queue_families = get_queue_families(vulkan, physical_device, surface);

    // Aggregate queue family indices and count the number of queues each family may need allocated.
    let mut family_map = SmallVec::<[u32; EXPECTED_MAX_QUEUE_FAMILIES]>::from_elem(
        0,
        queue_families.queue_families.len(),
    );
    let all_family_types = [
        &queue_families.graphics,
        &queue_families.compute,
        &queue_families.present,
    ];
    for family in all_family_types {
        for index in family {
            family_map[*index as usize] += 1;
        }
    }
    let priorities = vec![1.; queue_families.queue_families.len()];

    // Print a message if the queue family map has spilled over to the heap.
    #[cfg(debug_assertions)]
    if family_map.spilled() {
        println!(
            "INFO: Queue family map has spilled over to the heap. Family count {} greater than inline size {}",
            queue_families.queue_families.len(),
            family_map.inline_size(),
        );
    }

    // Describe the queue families that will be used with the new logical device.
    let queue_info = family_map
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| {
            if count == 0 {
                return None;
            }
            let priorities = &priorities[..count as usize];

            // Limit to the number of available queues. Guaranteed to be non-zero.
            let queue_count = count.min(queue_families.queue_families[index].queue_count);

            Some(ash::vk::DeviceQueueCreateInfo {
                queue_family_index: index as u32,
                queue_count,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            })
        })
        .collect::<Vec<_>>();

    // Create the logical device with the desired queue families and extensions.
    let device = unsafe {
        let device_info = ash::vk::DeviceCreateInfo {
            queue_create_info_count: queue_info.len() as u32,
            p_queue_create_infos: queue_info.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };
        vulkan
            .instance
            .create_device(physical_device, &device_info, None)
            .expect("Unable to create logical device")
    };

    (device, queue_families)
}

/// A helper for creating shader modules on a logical device.
pub fn create_shader_module(device: &ash::Device, code: &[u32]) -> ash::vk::ShaderModule {
    unsafe {
        device.create_shader_module(
            &ash::vk::ShaderModuleCreateInfo {
                code_size: code.len() << 2, // Expects the code size to be in bytes.
                p_code: code.as_ptr(),
                ..Default::default()
            },
            None,
        )
    }
    .expect("Unable to create shader module")
}

/// Check if the image format is for a depth buffer.
pub fn is_depth_format(format: ash::vk::Format) -> bool {
    use ash::vk::Format;
    matches!(
        format,
        Format::D16_UNORM
            | Format::D16_UNORM_S8_UINT
            | Format::D24_UNORM_S8_UINT
            | Format::D32_SFLOAT
            | Format::D32_SFLOAT_S8_UINT
            | Format::X8_D24_UNORM_PACK32
    )
}

/// Check if the image format is for a stencil buffer.
pub fn is_stencil_format(format: ash::vk::Format) -> bool {
    use ash::vk::Format;
    matches!(
        format,
        Format::S8_UINT
            | Format::D16_UNORM_S8_UINT
            | Format::D24_UNORM_S8_UINT
            | Format::D32_SFLOAT_S8_UINT
    )
}

/// Create a new image view for an existing Vulkan image with a specified format and MIP level.
pub fn create_image_view(
    device: &ash::Device,
    image: ash::vk::Image,
    format: ash::vk::Format,
    mip_levels: u32,
) -> ash::vk::ImageView {
    // Determine what kind of image view to create based on the format.
    let aspect_mask = if is_depth_format(format) {
        ash::vk::ImageAspectFlags::DEPTH
    } else if is_stencil_format(format) {
        ash::vk::ImageAspectFlags::STENCIL
    } else {
        ash::vk::ImageAspectFlags::COLOR
    };

    // Create the image view with the specified parameters.
    // TODO: More parameters will be needed when supporting VR and 3D images.
    let view_info = ash::vk::ImageViewCreateInfo {
        image,
        view_type: ash::vk::ImageViewType::TYPE_2D, // 2D image. Use 2D array for 3D images and VR.
        format,
        components: ash::vk::ComponentMapping::default(),
        subresource_range: ash::vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1, // In case of 3D images, `VK_REMAINING_ARRAY_LAYERS` can be used.
        },
        ..Default::default()
    };
    unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("Unable to create image view")
    }
}

/// Preferences for the image and behavior used with a new swapchain.
#[derive(Clone, Copy, Default)]
pub struct SwapchainPreferences {
    pub format: Option<ash::vk::Format>,
    pub color_space: Option<ash::vk::ColorSpaceKHR>,
    pub present_mode: Option<ash::vk::PresentModeKHR>,
}

pub struct Swapchain {
    swapchain_device: ash::khr::swapchain::Device,
    swapchain: ash::vk::SwapchainKHR,
    images: Vec<ash::vk::Image>,
    image_views: Vec<ash::vk::ImageView>,
    format: ash::vk::Format,
    color_space: ash::vk::ColorSpaceKHR,
    present_mode: ash::vk::PresentModeKHR,
    extent: ash::vk::Extent2D,
    image_available: ash::vk::Semaphore,
    image_rendered: ash::vk::Semaphore,
    acquire_fence: ash::vk::Fence,
    acquired_index: Option<u32>,
}

impl Swapchain {
    /// Create a new swapchain with the specified parameters and Vulkan instance.
    pub fn new(
        vulkan: &VulkanCore,
        physical_device: ash::vk::PhysicalDevice,
        logical_device: &ash::Device,
        surface: ash::vk::SurfaceKHR,
        preferences: SwapchainPreferences,
        old_swapchain: Option<ash::vk::SwapchainKHR>,
    ) -> Self {
        let surface_capabilities = unsafe {
            vulkan
                .khr
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Unable to get surface capabilities")
        };

        #[cfg(debug_assertions)]
        println!("Surface capabilities: {surface_capabilities:?}\n");

        let min_images = surface_capabilities.min_image_count;
        let max_images = NonZeroU32::new(surface_capabilities.max_image_count);

        // Determine which present modes are available.
        let supported_present_modes = unsafe {
            vulkan
                .khr
                .get_physical_device_surface_present_modes(physical_device, surface)
                .expect("Unable to get supported present modes")
        };

        #[cfg(debug_assertions)]
        println!("Supported present modes: {supported_present_modes:?}\n");

        // Default to what is guaranteed to be available.
        let preferred_present_mode = preferences
            .present_mode
            .unwrap_or(ash::vk::PresentModeKHR::FIFO);

        // Attempt to use the preferred present mode, or fallback to a default mode.
        // Choose the number of images based on the present mode and the minimum images required.
        let (present_mode, image_count) = supported_present_modes
            .iter()
            .find_map(|&mode| {
                // Don't choose anything other than the preferred present mode in this first pass.
                if mode != preferred_present_mode {
                    return None;
                }

                match preferred_present_mode {
                    // Immediate mode should use the minimum number of images supported.
                    // The caller has indicated they are not concerned with screen tearing, only resource usage.
                    ash::vk::PresentModeKHR::IMMEDIATE => Some((mode, min_images)),

                    // Mailbox mode requires at least 3 images for proper mailbox synchronization.
                    ash::vk::PresentModeKHR::MAILBOX => {
                        if max_images.unwrap_or(NonZeroU32::MAX)
                            >= unsafe { NonZeroU32::new_unchecked(3) }
                        {
                            // Ensure we are still using at least the minimum number of images.
                            Some((mode, min_images.max(3)))
                        } else {
                            None
                        }
                    }

                    // FIFO and FIFO_RELAXED modes require at least two images for proper vertical synchronization.
                    ash::vk::PresentModeKHR::FIFO | ash::vk::PresentModeKHR::FIFO_RELAXED => {
                        // Ensure we are still using at least the minimum number of images.
                        Some((mode, min_images.max(2)))
                    }

                    // No other named present modes currently exist, unknown how to handle them.
                    _ => None,
                }
            })
            .unwrap_or((ash::vk::PresentModeKHR::FIFO, min_images.max(2)));

        // Determine the image format that is supported and compare it to what is preferred.
        let supported_formats = unsafe {
            vulkan
                .khr
                .get_physical_device_surface_formats(physical_device, surface)
                .expect("Unable to get supported surface formats")
        };

        #[cfg(debug_assertions)]
        println!("Supported surface formats: {supported_formats:?}\n");

        let &ash::vk::SurfaceFormatKHR {
            format: image_format,
            color_space: image_color_space,
        } = {
            let lazy_is_color = |f| !is_depth_format(f) && !is_stencil_format(f);
            let color_space = preferences
                .color_space
                .unwrap_or(ash::vk::ColorSpaceKHR::SRGB_NONLINEAR);
            if let Some(format) = supported_formats.iter().find(|f| {
                preferences
                    .format
                    .map_or_else(|| lazy_is_color(f.format), |fmt| f.format == fmt)
                    && f.color_space == color_space
            }) {
                // Use the preferred format if it is supported.
                format
            } else {
                // Otherwise, use the first supported format that is neither a depth nor a stencil format.
                supported_formats
                    .iter()
                    .find(|f| lazy_is_color(f.format))
                    .expect("Unable to find a suitable image format")
            }
        };

        // Prefer the post-multiplied alpha composite alpha mode if available to allow blending when supported.
        let composite_alpha = if surface_capabilities
            .supported_composite_alpha
            .contains(ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
        {
            ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
        } else {
            ash::vk::CompositeAlphaFlagsKHR::OPAQUE
        };

        // In practice, window managers may set the surface to a zero extent when minimized.
        if surface_capabilities.max_image_extent.width == 0
            && surface_capabilities.max_image_extent.height == 0
        {
            panic!(
                "Surface capabilities have not been initialized or window has launched minimized"
            );
        }

        // Massage the current extent to ensure it is within the bounds of the surface.
        let extent = ash::vk::Extent2D {
            width: surface_capabilities
                .current_extent
                .width
                .max(surface_capabilities.min_image_extent.width)
                .min(surface_capabilities.max_image_extent.width),
            height: surface_capabilities
                .current_extent
                .height
                .max(surface_capabilities.min_image_extent.height)
                .min(surface_capabilities.max_image_extent.height),
        };

        // Create the swapchain with the specified parameters.
        let swapchain_info = ash::vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format,
            image_color_space,
            image_extent: extent,
            image_array_layers: 1, // Always 1 unless stereoscopic 3D (VR/AR) is used.
            image_usage: ash::vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: ash::vk::SharingMode::EXCLUSIVE, // Only one queue family will access the images.
            pre_transform: surface_capabilities.current_transform, // Do not apply additional transformation to the surface.
            composite_alpha,
            present_mode,
            clipped: ash::vk::TRUE, // Allow shaders to avoid updating regions that are obscured (by other windows, etc.)
            old_swapchain: old_swapchain.unwrap_or_default(),
            ..Default::default()
        };

        // Get swapchain-specific function pointers for this logical device.
        let swapchain_device = ash::khr::swapchain::Device::new(&vulkan.instance, logical_device);

        // Create the swapchain with the specified parameters.
        let swapchain = unsafe {
            swapchain_device
                .create_swapchain(&swapchain_info, None)
                .expect("Unable to create swapchain")
        };

        // Determine the actual number of images in the swapchain and create image views for each.
        let images = unsafe {
            swapchain_device
                .get_swapchain_images(swapchain)
                .expect("Unable to get swapchain images")
        };
        let image_views = images
            .iter()
            .map(|&i| create_image_view(logical_device, i, image_format, 1))
            .collect();

        // Create synchronization objects. Semaphores synchronize between different operations on the GPU; fences synchronize operations between the CPU and GPU.
        // TODO: The Vulkan tutorial published by the Khronos Group uses an image available and image rendered semaphore for each frame in flight. Not clear whether this is necessary, but should monitor.
        //       https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/03_Drawing/03_Frames_in_flight.html
        let image_available = unsafe {
            logical_device
                .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)
                .expect("Unable to create image available semaphore")
        };
        let image_rendered = unsafe {
            logical_device
                .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)
                .expect("Unable to create render finished semaphore")
        };
        let acquire_fence = unsafe {
            // Create the acquire fence in the signaled state to avoid waiting on the first acquire.
            logical_device
                .create_fence(
                    &ash::vk::FenceCreateInfo {
                        flags: ash::vk::FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    },
                    None,
                )
                .expect("Unable to create the acquire fence")
        };

        #[cfg(debug_assertions)]
        println!("New Swapchain: Present mode: {image_count} * {present_mode:?}: Format {image_format:?} in {image_color_space:?}\n");

        Self {
            swapchain_device,
            swapchain,
            images,
            image_views,
            format: image_format,
            color_space: image_color_space,
            present_mode,
            extent,
            image_available,
            image_rendered,
            acquire_fence,
            acquired_index: None,
        }
    }

    /// Delete the swapchain and its associated resources before dropping ownership.
    /// # Safety
    /// This function **must** only be called when the owned resources are not currently being processed by the GPU.
    pub fn destroy(
        self,
        logical_device: &ash::Device,
        framebuffers: &mut Vec<ash::vk::Framebuffer>,
    ) {
        // Destroy resources in the reverse order they were created.
        unsafe {
            logical_device
                .wait_for_fences(&[self.acquire_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Unable to wait for the acquire fence");
            logical_device.destroy_fence(self.acquire_fence, None);
            logical_device.destroy_semaphore(self.image_rendered, None);
            logical_device.destroy_semaphore(self.image_available, None);

            for framebuffer in framebuffers.drain(..) {
                logical_device.destroy_framebuffer(framebuffer, None);
            }

            for &image_view in &self.image_views {
                logical_device.destroy_image_view(image_view, None);
            }

            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
    }

    /// Recreate the swapchain using the existing one.
    /// This is useful when the window is resized, or the window is moved to a different monitor.
    /// # Notes
    /// * This function will wait for the logical device to finish its operations on the swapchain before recreating it.
    /// * This will
    pub fn recreate_swapchain(
        &mut self,
        vulkan: &VulkanCore,
        physical_device: ash::vk::PhysicalDevice,
        logical_device: &ash::Device,
        surface: ash::vk::SurfaceKHR,
        framebuffers: &mut Vec<ash::vk::Framebuffer>,
        preferences: SwapchainPreferences,
    ) {
        let mut stack_var_swapchain = Self::new(
            vulkan,
            physical_device,
            logical_device,
            surface,
            preferences,
            Some(self.swapchain),
        );

        // Wait for the logical device to finish its operations on the swapchain.
        unsafe {
            logical_device
                .device_wait_idle()
                .expect("Unable to wait for the logical device to finish its operations");
        }

        // Swap the original swapchain (`self`) with the new one.
        std::mem::swap(self, &mut stack_var_swapchain);

        // Destroy the old swapchain and its associated resources.
        stack_var_swapchain.destroy(logical_device, framebuffers);
    }

    /// Acquire the next image in the swapchain. Maintain the index of the acquired image.
    pub fn acquire_next_image(
        &mut self,
        device: &ash::Device,
    ) -> ash::prelude::VkResult<(ash::vk::ImageView, u32, bool)> {
        // Wait for the last acquire to complete its operation within the swapchain.
        unsafe {
            device
                .wait_for_fences(&[self.acquire_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Failed to wait for the acquire fence");
            device
                .reset_fences(&[self.acquire_fence])
                .expect("Failed to reset the acquire fence");
        }

        // Acquire the next image in the swapchain, using the same fence to signal completion.
        let (acquired_index, suboptimal) = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                FIVE_SECONDS_IN_NANOSECONDS,
                self.image_available,
                self.acquire_fence,
            )?
        };

        // Update our internal state with the acquired image's index.
        self.acquired_index = Some(acquired_index);

        Ok((
            self.image_views[acquired_index as usize],
            acquired_index,
            suboptimal,
        ))
    }

    /// Present the next image in the swapchain. Return whether the swapchain is suboptimal for the surface on success.
    pub fn present(&mut self, present_queue: ash::vk::Queue) -> ash::prelude::VkResult<bool> {
        let acquired_index = self
            .acquired_index
            .take()
            .expect("No image has been acquired by the swapchain before presenting");
        unsafe {
            self.swapchain_device.queue_present(
                present_queue,
                &ash::vk::PresentInfoKHR {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &self.image_rendered,
                    swapchain_count: 1,
                    p_swapchains: &self.swapchain,
                    p_image_indices: &acquired_index, // Needs to have the same number of entries as `swapchain_count`, i.e. 1.
                    ..Default::default()
                },
            )
        }
    }

    // Swapchain getters.
    pub fn extent(&self) -> ash::vk::Extent2D {
        self.extent
    }
    pub fn image_available(&self) -> ash::vk::Semaphore {
        self.image_available
    }
    pub fn image_count(&self) -> usize {
        self.images.len()
    }
    pub fn image_format(&self) -> ash::vk::Format {
        self.format
    }
    pub fn image_rendered(&self) -> ash::vk::Semaphore {
        self.image_rendered
    }
    pub fn image_views(&self) -> &[ash::vk::ImageView] {
        &self.image_views
    }
    pub fn present_mode(&self) -> ash::vk::PresentModeKHR {
        self.present_mode
    }
}
