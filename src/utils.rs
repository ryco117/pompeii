use std::{ffi::CStr, num::NonZeroU32};

use smallvec::SmallVec;
use strum::EnumCount as _;

/// The main Vulkan library interface.
pub struct VulkanCore {
    pub api: ash::Entry,
    pub instance: ash::Instance,
}

// A helper for scoring a device's dedication to graphics processing.
fn score_device_type(device_type: ash::vk::PhysicalDeviceType) -> u8 {
    match device_type {
        ash::vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        ash::vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        ash::vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

/// Try to find the best physical device for the application.
pub fn find_best_physical_device(
    instance: &ash::Instance,
    required_device_extensions: &[*const i8],
) -> Option<ash::vk::PhysicalDevice> {
    let mut physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Unable to enumerate physical devices")
    }
    .into_iter()
    .map(|device| unsafe { (device, instance.get_physical_device_properties(device)) })
    .collect::<Vec<_>>();

    // Sort the physical devices by the device type with preference for GPU's, then descending by graphics dedication.
    // If device types are equivalent, then sort by the maximum push constants size, since this is both practical and indicative of performance.
    physical_devices.sort_by(|(_, a), (_, b)| {
        // Sorting order is reversed to get the best device first.
        let device_type_cmp =
            score_device_type(b.device_type).cmp(&score_device_type(a.device_type));
        if device_type_cmp == std::cmp::Ordering::Equal {
            b.limits
                .max_push_constants_size
                .cmp(&a.limits.max_push_constants_size)
        } else {
            device_type_cmp
        }
    });

    // Find the first physical device that supports the required extensions and API.
    physical_devices
        .into_iter()
        .find_map(|(device, properties)| {
            if properties.api_version < ash::vk::API_VERSION_1_3 {
                return None;
            }

            let extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Unable to enumerate device extensions")
            };

            if required_device_extensions.iter().all(|&req| {
                let req = unsafe { CStr::from_ptr(req) };
                extensions.iter().any(|e| {
                    e.extension_name_as_c_str()
                        .expect("Available extension name is not a valid C string")
                        == req
                })
            }) {
                Some(device)
            } else {
                None
            }
        })
}

/// Available queue family types and their indices.
pub struct QueueFamilies {
    pub graphics: Option<u32>,
    pub compute: Option<u32>,
    pub present: Option<u32>,
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

    // Find the first queue families for each desired queue type.
    // Use an array with indices to allow compile-time guarantees about the number of queue types.
    // TODO: Consider maintaining a list of queue family indices for each type of queue.
    let mut all_queue_families = [None; QueueType::COUNT];
    for (family_index, queue_family) in queue_families.iter().enumerate() {
        if all_queue_families[QueueType::Graphics as usize].is_none()
            && queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::GRAPHICS)
        {
            all_queue_families[QueueType::Graphics as usize] = Some(family_index as u32);
        }

        if all_queue_families[QueueType::Compute as usize].is_none()
            && queue_family
                .queue_flags
                .contains(ash::vk::QueueFlags::COMPUTE)
        {
            all_queue_families[QueueType::Compute as usize] = Some(family_index as u32);
        }

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
    let mut family_map = SmallVec::<[u32; 16]>::from_elem(0, queue_families.len());
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

    // Describe the queue families that will be used.
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

    // Describe the device extensions that will be used, along with queue info.
    let device_info = ash::vk::DeviceCreateInfo {
        queue_create_info_count: queue_info.len() as u32,
        p_queue_create_infos: queue_info.as_ptr(),
        enabled_extension_count: device_extensions.len() as u32,
        pp_enabled_extension_names: device_extensions.as_ptr(),
        ..Default::default()
    };

    // Create the logical device with the desired queue families and extensions.
    let device = unsafe {
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
