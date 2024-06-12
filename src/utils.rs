use std::{ffi::CStr, num::NonZeroU32};

use smallvec::SmallVec;
use strum::EnumCount as _;

/// The main Vulkan library interface.
pub struct VulkanCore {
    pub api: ash::Entry,
    pub instance: ash::Instance,
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
        p_next: &mut ray_query as *mut _ as *mut std::ffi::c_void,
        ..Default::default()
    };
    let mut acceleration_structure = ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
        p_next: &mut ray_tracing as *mut _ as *mut std::ffi::c_void,
        ..Default::default()
    };
    let mut features = ash::vk::PhysicalDeviceFeatures2 {
        p_next: &mut acceleration_structure as *mut _ as *mut std::ffi::c_void,
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
            p_next: (&mut ray_tracing_pipeline) as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
        Some(ray_tracing_pipeline)
    } else {
        None
    }
}

/// Get all physical devices that support Vulkan 1.3 and the required extensions. Sort them by their likelyhood of being the desired device.
//  TODO: Investigate if `enumerate_physical_devices` can be replaced with something I can pass a `SmallVec` ref to.
pub fn get_physical_devices(
    instance: &ash::Instance,
    required_extensions: &[*const i8],
) -> SmallVec<[(ash::vk::PhysicalDevice, ash::vk::PhysicalDeviceProperties); 4]> {
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

            // Ensure every device has their properties printed in debug mode. Called before filtering starts.
            #[cfg(debug_assertions)]
            println!("Physical device {device:?}: {properties:?}");

            // Ensure the device supports Vulkan 1.3.
            if properties.api_version < ash::vk::API_VERSION_1_3 {
                return None;
            }

            // Query for the basic extensions of the physical device.
            let extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Unable to enumerate device extensions")
            };

            // Ensure the device supports all required extensions.
            if required_extensions.iter().all(|&req| {
                let req = unsafe { CStr::from_ptr(req) };
                extensions.iter().any(|e| {
                    e.extension_name_as_c_str()
                        .expect("Available extension name is not a valid C string")
                        == req
                })
            }) {
                Some((device, properties))
            } else {
                None
            }
        })
        .collect::<SmallVec<[(ash::vk::PhysicalDevice, ash::vk::PhysicalDeviceProperties); 4]>>();

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
    pub graphics: Option<u32>,
    pub compute: Option<u32>,
    pub present: Option<u32>,
    pub queue_families: Vec<ash::vk::QueueFamilyProperties>,
}

#[derive(strum_macros::EnumCount)]
#[repr(usize)]
enum QueueType {
    Graphics,
    Compute,
    Present,
}

/// Create a Vulkan logical device capable of graphics, compute, and presentation queues.
/// Returns the device and the queue family indices that were requested for use.
pub fn new_device(
    vulkan: &VulkanCore,
    physical_device: ash::vk::PhysicalDevice,
    surface: ash::vk::SurfaceKHR,
    device_extensions: &[*const i8],
) -> (ash::Device, QueueFamilies) {
    // Get the list of available queue families for this device.
    let queue_families = unsafe {
        vulkan
            .instance
            .get_physical_device_queue_family_properties(physical_device)
    };

    #[cfg(debug_assertions)]
    println!("Queue families: {queue_families:?}");

    // Find the first queue families for each desired queue type.
    // Use an array with indices to allow compile-time guarantees about the number of queue types.
    // TODO: Consider maintaining a list of queue family indices for each type of queue.
    //       This will also make it easier to put different queue types on different families if they are available.
    // TODO: Make this logic its own function.
    let mut all_queue_families = [None; QueueType::COUNT];
    for (family_index, queue_family) in queue_families.iter().enumerate() {
        // Get a present queue family.
        if all_queue_families[QueueType::Present as usize].is_none() {
            let is_present_supported = unsafe {
                ash::khr::surface::Instance::new(&vulkan.api, &vulkan.instance)
                    .get_physical_device_surface_support(
                        physical_device,
                        family_index as u32,
                        surface,
                    )
                    .expect("Unable to check if 'present' is supported")
            };
            if is_present_supported {
                all_queue_families[QueueType::Present as usize] = Some(family_index as u32);
            }
        }

        // Get a graphics queue family.
        if all_queue_families[QueueType::Graphics as usize].is_none()
            && queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::GRAPHICS)
        {
            all_queue_families[QueueType::Graphics as usize] = Some(family_index as u32);
        }

        // Get a compute queue family.
        if all_queue_families[QueueType::Compute as usize].is_none()
            && queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::COMPUTE)
        {
            all_queue_families[QueueType::Compute as usize] = Some(family_index as u32);
        }

        // Break early if all families are found.
        if all_queue_families.iter().all(Option::is_some) {
            break;
        }
    }

    // TODO: Allow the caller to choose which queue types are needed.
    let graphics_family = all_queue_families[QueueType::Graphics as usize]
        .expect("Unable to find a graphics queue family");
    let compute_family = all_queue_families[QueueType::Compute as usize]
        .expect("Unable to find a compute queue family");
    let present_family = all_queue_families[QueueType::Present as usize]
        .expect("Unable to find a present queue family");

    // Aggregate queue family indices and count the number of queues each family may need allocated.
    let mut family_map = SmallVec::<[u32; 8]>::from_elem(0, queue_families.len());
    let all_family_types = [graphics_family, compute_family, present_family];
    for family in all_family_types {
        family_map[family as usize] += 1;
    }
    let priorities = [1.; QueueType::COUNT];

    // Print a message if the queue family map has spilled over to the heap.
    #[cfg(debug_assertions)]
    if family_map.spilled() {
        println!(
            "INFO: Queue family map has spilled over to the heap. Family count {} greater than inline size {}",
            queue_families.len(),
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
            Some(ash::vk::DeviceQueueCreateInfo {
                queue_family_index: index as u32,
                queue_count: count.min(queue_families[index].queue_count), // Limit to the number of available queues.
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

    (
        device,
        QueueFamilies {
            graphics: Some(graphics_family),
            compute: Some(compute_family),
            present: Some(present_family),
            queue_families,
        },
    )
}

pub struct Swapchain {
    swapchain: ash::vk::SwapchainKHR,
    pub images: Vec<ash::vk::Image>,
    pub image_views: Vec<ash::vk::ImageView>,
    pub format: ash::vk::Format,
    pub extent: ash::vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        vulkan: &VulkanCore,
        physical_device: ash::vk::PhysicalDevice,
        logical_device: &ash::Device,
        surface: ash::vk::SurfaceKHR,
        preferred_present_mode: ash::vk::PresentModeKHR,
    ) -> ash::vk::SwapchainKHR {
        let khr_instance = ash::khr::surface::Instance::new(&vulkan.api, &vulkan.instance);
        let surface_capabilities = unsafe {
            khr_instance
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Unable to get surface capabilities")
        };

        let min_images = surface_capabilities.min_image_count;
        let max_images = NonZeroU32::new(surface_capabilities.max_image_count);

        // Determine which present modes are available.
        let supported_present_modes = unsafe {
            khr_instance
                .get_physical_device_surface_present_modes(physical_device, surface)
                .expect("Unable to get supported present modes")
        };

        // Attempt to use the preferred present mode, or fallback to a default mode.
        // Choose the number of images based on the present mode and the minimum images required.
        let (present_mode, image_count) = supported_present_modes
            .iter()
            .find_map(|&mode| {
                if mode == preferred_present_mode {
                    match preferred_present_mode {
                        // Immediate mode should use the minimum number of images supported.
                        ash::vk::PresentModeKHR::IMMEDIATE => Some((mode, min_images)),

                        // Mailbox mode requires at least 3 images.
                        ash::vk::PresentModeKHR::MAILBOX => {
                            if max_images.unwrap_or(NonZeroU32::MAX)
                                >= unsafe { NonZeroU32::new_unchecked(3) }
                            {
                                Some((mode, min_images.max(3)))
                            } else {
                                None
                            }
                        }

                        // FIFO and FIFO_RELAXED modes require at least
                        ash::vk::PresentModeKHR::FIFO | ash::vk::PresentModeKHR::FIFO_RELAXED => {
                            Some((mode, min_images.max(2)))
                        }

                        // More exotic present mode.
                        // TODO: Figure out how many images are needed for each other mode.
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or((ash::vk::PresentModeKHR::FIFO, min_images.max(2)));

        let swapchain_info = ash::vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format: ash::vk::Format::B8G8R8A8_SRGB,
            image_color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
            image_extent: surface_capabilities.current_extent, // Platform dependent behavior if these values don't match.
            image_array_layers: 1, // Always 1 unless stereoscopic 3D is used.
            image_usage: ash::vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: ash::vk::SharingMode::EXCLUSIVE, // Only one queue family will access the images.
            pre_transform: surface_capabilities.current_transform, // No transform.
            composite_alpha: ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED, // Blend with other windows.
            present_mode,
            clipped: ash::vk::TRUE, // Allow shaders to avoid updating obscured regions.
            ..Default::default()
        };

        unsafe {
            let swap_device = ash::khr::swapchain::Device::new(&vulkan.instance, &logical_device);
            swap_device
                .create_swapchain(&swapchain_info, None)
                .expect("Unable to create swapchain")
        }
    }
}
