use std::{num::NonZeroU32, sync::Mutex};

use smallvec::SmallVec;

use super::{
    create_image, create_image_view, is_depth_format, is_stencil_format, query_multisample_support,
    MultiSampleAntiAliasing, VulkanCore, EXPECTED_MAX_FRAMES_IN_FLIGHT,
    FIVE_SECONDS_IN_NANOSECONDS,
};

/// Vulkan uses this "special value" to indicate that the application must set a desired extent without a `current_extent` available.
/// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceCapabilitiesKHR.html>
pub const SPECIAL_SURFACE_EXTENT: ash::vk::Extent2D = ash::vk::Extent2D {
    width: u32::MAX,
    height: u32::MAX,
};

/// The default extent to use when the surface `current_extent` is `SPECIAL_SURFACE_EXTENT` and no
/// preferred extent is available.
pub const DEFAULT_SURFACE_EXTENT: ash::vk::Extent2D = ash::vk::Extent2D {
    width: 800,
    height: 600,
};

/// Preferences for the image and behavior used with a new swapchain.
#[derive(Clone, Copy, Default)]
pub struct SwapchainPreferences {
    pub format: Option<ash::vk::Format>,
    pub color_space: Option<ash::vk::ColorSpaceKHR>,
    pub present_mode: Option<ash::vk::PresentModeKHR>,
    pub color_samples: Option<ash::vk::SampleCountFlags>,

    /// The preferred extent to use if and only if the surface wants the caller to specify an extent to use.
    pub preferred_extent: Option<ash::vk::Extent2D>,
}

/// Synchronization objects for a frame in flight.
pub struct FrameInFlightSync {
    pub image_available: ash::vk::Semaphore,
    pub image_rendered: ash::vk::Semaphore,
    pub present_complete: ash::vk::Fence,
}

/// A Vulkan swapchain with synchronization objects for each frame in flight.
pub struct Swapchain {
    swapchain_device: ash::khr::swapchain::Device,
    handle: ash::vk::SwapchainKHR,
    image_views: Vec<ash::vk::ImageView>,
    images: Vec<ash::vk::Image>,
    format: ash::vk::Format,
    color_space: ash::vk::ColorSpaceKHR,
    present_mode: ash::vk::PresentModeKHR,
    extent: ash::vk::Extent2D,
    frame_syncs: SmallVec<[FrameInFlightSync; EXPECTED_MAX_FRAMES_IN_FLIGHT]>,
    current_frame: usize,
    acquired_index: Option<u32>,
    multisample: Option<MultiSampleAntiAliasing>,
    enabled_swapchain_maintenance1: bool,
}

/// The result of acquiring the next image from the swapchain and advancing to the next frame in flight.
pub struct NextSwapchainImage {
    pub image_index: u32,
    pub suboptimal: bool,
}

impl Swapchain {
    /// Create a new swapchain with the specified parameters and Vulkan instance.
    /// # Panics
    /// * The `VulkanCore` struct must have a `khr` field that is not `None`.
    pub fn new(
        vulkan: &VulkanCore,
        physical_device: ash::vk::PhysicalDevice,
        logical_device: &ash::Device,
        surface: ash::vk::SurfaceKHR,
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        image_usage: Option<ash::vk::ImageUsageFlags>,
        preferences: SwapchainPreferences,
        enabled_swapchain_maintenance1: bool,
        old_swapchain: Option<ash::vk::SwapchainKHR>,
    ) -> Self {
        let khr = vulkan
            .surface_instance
            .as_ref()
            .expect("Vulkan instance does not support the KHR surface instance extension");
        let surface_capabilities = unsafe {
            khr.get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Unable to get surface capabilities")
        };

        #[cfg(debug_assertions)]
        println!("INFO: Surface capabilities: {surface_capabilities:?}\n");

        // Try to choose the preferred present mode, but fall back to the default of FIFO. Uses the most reasonable and valid image count for each present mode.
        let (present_mode, image_count) = Self::choose_present_mode_and_image_count(
            vulkan,
            physical_device,
            surface,
            &surface_capabilities,
            preferences.present_mode,
        );

        // Determine the image format that is supported and compare it to what is preferred.
        let supported_formats = unsafe {
            khr.get_physical_device_surface_formats(physical_device, surface)
                .expect("Unable to get supported surface formats")
        };

        #[cfg(debug_assertions)]
        println!("INFO: Supported surface formats: {supported_formats:?}\n");

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

        let extent = if surface_capabilities.current_extent == SPECIAL_SURFACE_EXTENT {
            preferences
                .preferred_extent
                .unwrap_or(DEFAULT_SURFACE_EXTENT)
        } else {
            // In practice, window managers may set the surface to a zero extent when minimized.
            if surface_capabilities.max_image_extent.width == 0
                || surface_capabilities.max_image_extent.height == 0
            {
                panic!(
                    "Surface capabilities have not been initialized or window has launched minimized"
                );
            }
            surface_capabilities.current_extent
        };

        #[cfg(debug_assertions)]
        println!("INFO: Swapchain extent: {extent:?}\n");

        // Optionally provide additional present modes if the
        let mut present_modes_ext = if enabled_swapchain_maintenance1 {
            Some(ash::vk::SwapchainPresentModesCreateInfoEXT {
                present_mode_count: 1,
                p_present_modes: &present_mode,
                ..Default::default()
            })
        } else {
            None
        };

        // Create the swapchain with the specified parameters.
        let mut swapchain_info = ash::vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format,
            image_color_space,
            image_extent: extent,
            image_array_layers: 1, // Always 1 unless stereoscopic-3D / XR is used.
            image_usage: image_usage.unwrap_or(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT),
            image_sharing_mode: ash::vk::SharingMode::EXCLUSIVE, // Only one queue family will access the images.
            pre_transform: surface_capabilities.current_transform, // Do not apply additional transformation to the surface.
            composite_alpha,
            present_mode,
            clipped: ash::vk::TRUE, // Allow shaders to avoid updating regions that are obscured (by other windows, etc.)
            old_swapchain: old_swapchain.unwrap_or_default(),
            ..Default::default()
        };
        if let Some(present_modes_ext) = &mut present_modes_ext {
            swapchain_info = swapchain_info.push_next(present_modes_ext);
        }

        // Get swapchain-specific function pointers for this logical device.
        let swapchain_device = ash::khr::swapchain::Device::new(&vulkan.instance, logical_device);

        // Create the swapchain with the specified parameters.
        let swapchain = unsafe {
            swapchain_device
                .create_swapchain(&swapchain_info, None)
                .expect("Unable to create swapchain")
        };

        // Determine the actual number of images in the swapchain and create image views for each.
        let swapchain_images = unsafe {
            swapchain_device
                .get_swapchain_images(swapchain)
                .expect("Unable to get swapchain images")
        };

        // Determine if the caller is trying to use multiple color samples, and if it is supported.
        let multisample = if let Some(multisample_image_create) = query_multisample_support(
            vulkan,
            physical_device,
            preferences
                .color_samples
                .unwrap_or(ash::vk::SampleCountFlags::TYPE_1),
            image_format,
            extent,
            1,
            ash::vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | ash::vk::ImageUsageFlags::COLOR_ATTACHMENT, // Ensure the multisampled image is optimized to be transient.
        ) {
            let multisample_images: Vec<_> = swapchain_images
                .iter()
                .map(|_| {
                    create_image(
                        logical_device,
                        memory_allocator,
                        &multisample_image_create,
                        "Multisample Image",
                    )
                })
                .collect();
            let image_views = multisample_images
                .iter()
                .map(|(i, _)| create_image_view(logical_device, *i, image_format, 1))
                .collect();

            Some(MultiSampleAntiAliasing {
                samples: multisample_image_create.samples,
                images: multisample_images,
                image_views,
            })
        } else {
            None
        };

        // Create image views for each image in the swapchain.
        let swapchain_views = swapchain_images
            .iter()
            .map(|&i| create_image_view(logical_device, i, image_format, 1))
            .collect();

        // Create synchronization objects. Semaphores synchronize between different operations on the GPU; fences synchronize operations between the CPU and GPU.
        // We will have a frame in flight for each image in the swapchain, and at least two so that a new command can be recorded while another is read.
        let frames_in_flight = image_count.max(2) as usize;
        let frame_syncs = std::iter::repeat_with(|| {
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
            let present_complete = unsafe {
                logical_device
                    .create_fence(
                        &ash::vk::FenceCreateInfo {
                            flags: ash::vk::FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                        None,
                    )
                    .expect("Unable to create present complete fence")
            };
            FrameInFlightSync {
                image_available,
                image_rendered,
                present_complete,
            }
        })
        .take(frames_in_flight)
        .collect();

        #[cfg(debug_assertions)]
        println!("INFO: New Swapchain: Present mode: {image_count} * {present_mode:?}: Format {image_format:?} in {image_color_space:?}\n");

        Self {
            swapchain_device,
            handle: swapchain,
            image_views: swapchain_views,
            images: swapchain_images,
            format: image_format,
            color_space: image_color_space,
            present_mode,
            extent,
            frame_syncs,
            current_frame: 0,
            acquired_index: None,
            multisample,
            enabled_swapchain_maintenance1,
        }
    }

    /// Delete the swapchain and its associated resources before dropping ownership.
    /// # Safety
    /// This function **must** only be called when the owned resources are not currently being processed by the GPU.
    pub fn destroy(
        self,
        logical_device: &ash::Device,
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
    ) {
        // Destroy resources in the reverse order they were created.
        unsafe {
            // Destroy the synchronization objects.
            for sync in self.frame_syncs {
                logical_device.destroy_semaphore(sync.image_available, None);
                logical_device.destroy_semaphore(sync.image_rendered, None);
                logical_device.destroy_fence(sync.present_complete, None);
            }

            // Destroy the image views and images.
            // NOTE: Do not directly destroy the images managed by the swapchain internally (i.e., the presentation images).
            for image_view in self.image_views {
                logical_device.destroy_image_view(image_view, None);
            }
            if let Some(multisample) = self.multisample {
                for image_view in multisample.image_views {
                    logical_device.destroy_image_view(image_view, None);
                }
                for (image, allocation) in multisample.images {
                    logical_device.destroy_image(image, None);
                    memory_allocator
                        .lock()
                        .expect("Failed to lock allocator in `Swapchain::destroy`")
                        .free(allocation)
                        .expect("Unable to free multisample image allocation");
                }
            }
            self.swapchain_device.destroy_swapchain(self.handle, None);
        }
    }

    /// Helper to attempt to usee the preferred present mode, but falls back to the default of `FIFO` which is always supported.
    /// Also, tries to use the most reasonable and valid image count for whichever present mode is determined.
    /// # Panics
    /// * The `utils::VulkanCore` struct must have a `khr` field that is not `None`.
    pub fn choose_present_mode_and_image_count(
        vulkan: &VulkanCore,
        physical_device: ash::vk::PhysicalDevice,
        surface: ash::vk::SurfaceKHR,
        surface_capabilities: &ash::vk::SurfaceCapabilitiesKHR,
        preferred_present_mode: Option<ash::vk::PresentModeKHR>,
    ) -> (ash::vk::PresentModeKHR, u32) {
        // NOTE: `SurfaceCapabilitiesKHR` specifies the minimum and maximum number of images that any present mode on this surface may have.
        // However, each present mode may have a tighter bound on the min and max than this global value.
        // See below where `SurfaceCapabilities2KHR` is used to get the actual min and max image count for the determined present mode.
        let surface_min_images = surface_capabilities.min_image_count;
        let surface_max_images = NonZeroU32::new(surface_capabilities.max_image_count);

        // Determine which present modes are available.
        let supported_present_modes = unsafe {
            vulkan
                .surface_instance
                .as_ref()
                .unwrap()
                .get_physical_device_surface_present_modes(physical_device, surface)
                .expect("Unable to get supported present modes")
        };

        #[cfg(debug_assertions)]
        println!("INFO: Supported present modes: {supported_present_modes:?}\n");

        // Default to what is guaranteed to be available.
        let preferred_present_mode =
            preferred_present_mode.unwrap_or(ash::vk::PresentModeKHR::FIFO);

        // Attempt to use the preferred present mode, or fallback to a default mode.
        // Choose the number of images based on the present mode and the minimum images required.
        let (present_mode, mut image_count) = supported_present_modes
            .iter()
            .find_map(|&mode| {
                // Don't choose anything other than the preferred present mode in this first pass.
                if mode != preferred_present_mode {
                    return None;
                }

                match preferred_present_mode {
                    // Immediate mode should use the minimum number of images supported.
                    // This mode is used when there is not a concern with screen tearing, only resource usage.
                    ash::vk::PresentModeKHR::IMMEDIATE => Some((mode, 1)),

                    // Use `MAILBOX` to reduce latency and avoid tearing. Generally preferred.
                    // `FIFO` and `FIFO_RELAXED` modes require at least two images for proper vertical synchronization.
                    ash::vk::PresentModeKHR::MAILBOX
                    | ash::vk::PresentModeKHR::FIFO
                    | ash::vk::PresentModeKHR::FIFO_RELAXED => Some((mode, 3)),

                    // No other named present modes currently exist so we default to FIFO.
                    _ => None,
                }
            })
            .unwrap_or((ash::vk::PresentModeKHR::FIFO, 3));

        if vulkan.enabled_instance_extension(ash::ext::surface_maintenance1::NAME) {
            let mut present_mode_ext =
                ash::vk::SurfacePresentModeEXT::default().present_mode(present_mode);
            let surface_info = ash::vk::PhysicalDeviceSurfaceInfo2KHR::default()
                .surface(surface)
                .push_next(&mut present_mode_ext);
            let mut surface_capabilities = ash::vk::SurfaceCapabilities2KHR::default();
            unsafe {
                ash::khr::get_surface_capabilities2::Instance::new(&vulkan.api, &vulkan.instance)
                    .get_physical_device_surface_capabilities2(
                        physical_device,
                        &surface_info,
                        &mut surface_capabilities,
                    )
            }
            .expect("Unable to get extended surface capabilities(2)");

            let present_max =
                NonZeroU32::new(surface_capabilities.surface_capabilities.max_image_count);
            image_count = image_count.clamp(
                surface_capabilities.surface_capabilities.min_image_count,
                present_max.map_or(u32::MAX, NonZeroU32::get),
            );
        } else {
            image_count = image_count.clamp(
                surface_min_images,
                surface_max_images.map_or(u32::MAX, NonZeroU32::get),
            );
        }

        (present_mode, image_count)
    }

    /// Recreate the swapchain using the existing one.
    /// This is useful when the window is resized, or the window is moved to a different monitor.
    /// # Notes
    /// * This function will wait for the logical device to finish its operations on the swapchain before recreating it.
    /// * The framebuffers will be destroyed and must be recreated by the caller.
    pub fn recreate_swapchain(
        &mut self,
        vulkan: &VulkanCore,
        physical_device: ash::vk::PhysicalDevice,
        logical_device: &ash::Device,
        surface: ash::vk::SurfaceKHR,
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        image_usage: Option<ash::vk::ImageUsageFlags>,
        preferences: SwapchainPreferences,
    ) {
        let mut stack_var_swapchain = Self::new(
            vulkan,
            physical_device,
            logical_device,
            surface,
            memory_allocator,
            image_usage,
            preferences,
            self.enabled_swapchain_maintenance1,
            Some(self.handle),
        );

        // Swap the original swapchain (`self`) with the new one.
        std::mem::swap(self, &mut stack_var_swapchain);
        let old_swapchain = stack_var_swapchain; // Variable rename for clarity.

        unsafe {
            if self.enabled_swapchain_maintenance1 {
                // Wait for the old swapchain to complete its presentation fences.
                let presentation_fences: SmallVec<[ash::vk::Fence; EXPECTED_MAX_FRAMES_IN_FLIGHT]> =
                    old_swapchain
                        .frame_syncs
                        .iter()
                        .map(|s| s.present_complete)
                        .collect();
                logical_device
                    .wait_for_fences(
                        presentation_fences.as_slice(),
                        true,
                        FIVE_SECONDS_IN_NANOSECONDS,
                    )
                    .expect("Unable to wait for the logical device to finish its operations");
            } else {
                // Wait for the logical device to finish its operations on the swapchain.
                // This is not particularly optimal.
                logical_device
                    .device_wait_idle()
                    .expect("Unable to wait for the logical device to finish its operations");
            }
        }

        // Destroy the old swapchain and its associated resources.
        old_swapchain.destroy(logical_device, memory_allocator);
    }

    /// Acquire the next image in the swapchain. Maintain the index of the acquired image.
    pub fn acquire_next_image(&mut self) -> ash::prelude::VkResult<NextSwapchainImage> {
        // Acquire the next image in the swapchain, using the same fence to signal completion.
        let (acquired_index, suboptimal) = unsafe {
            self.swapchain_device.acquire_next_image(
                self.handle,
                FIVE_SECONDS_IN_NANOSECONDS,
                self.image_available(),
                ash::vk::Fence::null(),
            )?
        };

        // Update our internal state with the acquired image's index.
        self.acquired_index = Some(acquired_index);

        Ok(NextSwapchainImage {
            image_index: acquired_index,
            suboptimal,
        })
    }

    /// Present the next image in the swapchain. Return whether the swapchain is suboptimal for the surface on success.
    pub fn present(
        &mut self,
        present_queue: ash::vk::Queue,
        use_present_fence: bool,
    ) -> ash::prelude::VkResult<bool> {
        let acquired_index = self
            .acquired_index
            .take()
            .expect("No image has been acquired by the swapchain before presenting");

        // Optionally, use a present fence to signal completion of the presentation operation.
        // This is only present with device extension `VK_EXT_swapchain_maintenance1`.
        let fence_info = if use_present_fence {
            Some(ash::vk::SwapchainPresentFenceInfoEXT {
                swapchain_count: 1,
                p_fences: &self.frame_syncs[self.current_frame].present_complete,
                ..Default::default()
            })
        } else {
            None
        };

        let result = unsafe {
            self.swapchain_device.queue_present(
                present_queue,
                &ash::vk::PresentInfoKHR {
                    p_next: fence_info
                        .as_ref()
                        .map_or(std::ptr::null(), |f| std::ptr::from_ref(f).cast()),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &self.image_rendered(),
                    swapchain_count: 1,
                    p_swapchains: &self.handle,
                    p_image_indices: &acquired_index, // Needs to have the same number of entries as `swapchain_count`, i.e. 1.
                    ..Default::default()
                },
            )
        };

        // Advance the current frame index to the next after successfully submitting the presentation command.
        self.current_frame = (self.current_frame + 1) % self.frame_syncs.len();

        result
    }

    // Swapchain getters.
    pub fn color_space(&self) -> ash::vk::ColorSpaceKHR {
        self.color_space
    }
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }
    pub fn extent(&self) -> ash::vk::Extent2D {
        self.extent
    }
    pub fn frames_in_flight(&self) -> usize {
        self.frame_syncs.len()
    }
    pub fn frame_syncs(&self) -> &[FrameInFlightSync] {
        &self.frame_syncs
    }
    pub fn image_available(&self) -> ash::vk::Semaphore {
        self.frame_syncs[self.current_frame].image_available
    }
    pub fn image_format(&self) -> ash::vk::Format {
        self.format
    }
    pub fn image_rendered(&self) -> ash::vk::Semaphore {
        self.frame_syncs[self.current_frame].image_rendered
    }
    pub fn image_views(&self) -> &[ash::vk::ImageView] {
        &self.image_views
    }
    pub fn images(&self) -> &[ash::vk::Image] {
        &self.images
    }
    pub fn multisample_count(&self) -> Option<ash::vk::SampleCountFlags> {
        self.multisample.as_ref().map(|m| m.samples)
    }
    pub fn multisample_views(&self) -> Option<&[ash::vk::ImageView]> {
        self.multisample.as_ref().map(|m| m.image_views.as_slice())
    }
    pub fn present_complete(&self) -> ash::vk::Fence {
        self.frame_syncs[self.current_frame].present_complete
    }
    pub fn present_mode(&self) -> ash::vk::PresentModeKHR {
        self.present_mode
    }
}
