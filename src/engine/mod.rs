use std::{collections::HashSet, ffi::CStr};

use smallvec::{smallvec, SmallVec};
use utils::{fxaa_pass::FxaaPass, EXPECTED_MAX_FRAMES_IN_FLIGHT, FIVE_SECONDS_IN_NANOSECONDS};

pub mod example_fluid;
pub mod example_triangle;
pub mod utils;

/// A sane constant for the expected maximum number of enabled device extensions. This is not for restrictions but to allow optimizations to avoid heap allocation.
const EXPECTED_MAX_ENABLED_DEVICE_EXTENSIONS: usize = 4;

/// The demos the application is capable of rendering.
pub enum DemoPipeline {
    Triangle(example_triangle::Pipeline),
    Fluid(example_fluid::FluidSimulation),
}

/// The push constants necessary to render the active demo.
/// Each demo needs a unique set of information to render each frame.
pub enum DemoPushConstants {
    Triangle(example_triangle::PushConstants),
    Fluid(example_fluid::PushConstants),
}

/// Define which rendering objects are necessary for this application.
pub struct Renderer {
    pub physical_device: ash::vk::PhysicalDevice,
    pub device_extensions: HashSet<&'static CStr>,
    pub logical_device: ash::Device,
    pageable_device_local_memory: Option<ash::ext::pageable_device_local_memory::Device>,

    pub surface: ash::vk::SurfaceKHR,
    memory_allocator: gpu_allocator::vulkan::Allocator,
    pub swapchain: utils::Swapchain,

    // The specific object we are interested in rendering.
    pub active_demo: DemoPipeline,

    graphics_queue: utils::IndexedQueue,
    compute_queue: utils::IndexedQueue,
    presentation_queue: utils::IndexedQueue,
    command_pool: ash::vk::CommandPool,
    compute_command_pool: Option<(ash::vk::CommandPool, ash::vk::Semaphore)>, // Optional compute command pool and compute semaphore if the graphics and compute queue families are separate.

    command_buffers: Vec<ash::vk::CommandBuffer>,
    frame_fences: Vec<ash::vk::Fence>,

    fxaa_pass: Option<FxaaPass>,
    pub swapchain_preferences: utils::SwapchainPreferences,
}

#[derive(Clone, Copy, Debug)]
pub enum DemoSpecializationConstants {
    Triangle(example_triangle::SpecializationConstants),

    #[allow(dead_code)]
    Fluid,
}

/// The new demo to switch to, and any unique parameters necessary to initialize it.
#[derive(Clone, Copy, Debug)]
pub enum NewDemo {
    Triangle(example_triangle::SpecializationConstants),
    Fluid,
}

impl Renderer {
    /// Create a new renderer for the application.
    pub fn new(
        vulkan: &utils::VulkanCore,
        surface: ash::vk::SurfaceKHR,
        swapchain_preferences: utils::SwapchainPreferences,
        specialization_constants: DemoSpecializationConstants,
        enable_fxaa: bool,
    ) -> Self {
        // Required device extensions for the swapchain.
        const DEVICE_EXTENSIONS: [*const i8; 1] = [ash::khr::swapchain::NAME.as_ptr()];

        // Define the device features that are required from this application.
        let required_device_features = utils::EnginePhysicalDeviceFeatures {
            buffer_device_address: ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
                .buffer_device_address(true),
            dynamic_rendering: ash::vk::PhysicalDeviceDynamicRenderingFeatures::default()
                .dynamic_rendering(true),
            synchronization2: ash::vk::PhysicalDeviceSynchronization2Features::default()
                .synchronization2(true),
            ..Default::default()
        };

        // Use simple heuristics to find the best suitable physical device.
        let (physical_device, device_properties, mut device_features) =
            *utils::get_sorted_physical_devices(
                &vulkan.instance,
                vulkan.version,
                &DEVICE_EXTENSIONS,
                &required_device_features,
            )
            .first()
            .expect("Unable to find a suitable physical device");

        #[cfg(debug_assertions)]
        println!(
            "Selected physical device: {:?}\n Features: {device_features:?}",
            device_properties
                .device_name_as_c_str()
                .expect("Unable to get device name")
        );

        // Check the physical device for optional features we can enable for this application.
        let mut custom_extensions =
            SmallVec::<[*const i8; EXPECTED_MAX_ENABLED_DEVICE_EXTENSIONS]>::new();
        let available_extensions = unsafe {
            vulkan
                .instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Unable to enumerate device extensions")
        };
        let enabled_swapchain_maintenance =
            if vulkan.enabled_instance_extension(ash::ext::surface_maintenance1::NAME) {
                if utils::extensions_list_contains(
                    &available_extensions,
                    ash::ext::swapchain_maintenance1::NAME,
                ) {
                    #[cfg(debug_assertions)]
                    println!("INFO: Enabling VK_EXT_swapchain_maintenance1 device extension");
                    custom_extensions.push(ash::ext::swapchain_maintenance1::NAME.as_ptr());

                    true
                } else {
                    false
                }
            } else {
                false
            };
        let enabled_pageable_device_local_memory = if utils::extensions_list_contains(
            &available_extensions,
            ash::ext::memory_priority::NAME,
        ) && utils::extensions_list_contains(
            &available_extensions,
            ash::ext::pageable_device_local_memory::NAME,
        ) {
            #[cfg(debug_assertions)]
            println!("INFO: Enabling VK_EXT_memory_priority device extension");
            custom_extensions.push(ash::ext::memory_priority::NAME.as_ptr());
            #[cfg(debug_assertions)]
            println!("INFO: Enabling VK_EXT_pageable_device_local_memory device extension");
            custom_extensions.push(ash::ext::pageable_device_local_memory::NAME.as_ptr());
            true
        } else {
            false
        };

        // We required these features in physical device selection.
        // Now we can request the features be enabled over the logical device we create.
        // Ignore the features we are not interested in, and enable the ones we are.
        let enabled_pageable_device_local_memory =
            enabled_pageable_device_local_memory && device_features.pageable_device_local_memory();
        let mut features = ash::vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut device_features.dynamic_rendering)
            .push_next(&mut device_features.synchronization2)
            .push_next(&mut device_features.buffer_device_address);

        // Add optional device features when they are available.
        if enabled_pageable_device_local_memory {
            #[cfg(debug_assertions)]
            println!("INFO: Enabling VK_EXT_pageable_device_local_memory device feature");

            features = features.push_next(&mut device_features.pageable_device_local_memory);
        }

        // Collect all the device extensions we need to enable.
        let all_extension_pointers = DEVICE_EXTENSIONS
            .iter()
            .chain(custom_extensions.iter())
            .copied()
            .collect::<Vec<_>>();

        // Create a logical device capable of rendering to the surface and performing compute operations.
        let (logical_device, queue_families) = utils::new_device(
            vulkan,
            physical_device,
            surface,
            &all_extension_pointers,
            Some(&mut features),
        );

        let pageable_device_local_memory = if enabled_pageable_device_local_memory {
            Some(ash::ext::pageable_device_local_memory::Device::new(
                &vulkan.instance,
                &logical_device,
            ))
        } else {
            None
        };

        let (graphics_index, compute_index, present_index) = {
            // NOTE: Prefer that the graphics and compute queues are equivalent because the `example_fluid` module will benefit from shared resources.
            let graphics = utils::get_queue_family_index(
                ash::vk::QueueFlags::GRAPHICS,
                ash::vk::QueueFlags::COMPUTE,
                &queue_families,
            );

            // Prefer using the graphics queue for presentation if possible.
            let present = if queue_families.present.contains(&graphics) {
                graphics
            } else {
                #[cfg(debug_assertions)]
                eprintln!(
                    "WARN: Using different queue families for presentation and primary graphics"
                );

                // Choose the first queue family that supports presentation.
                *queue_families
                    .present
                    .first()
                    .expect("Unable to find a present queue family")
            };

            // NOTE: Prefer that the graphics and compute queues are equivalent because the `example_fluid` module will benefit from shared resources.
            let compute = if queue_families.compute.contains(&graphics) {
                graphics
            } else {
                *queue_families
                    .compute
                    .first()
                    .expect("Unable to find a compute queue family")
            };

            (graphics, compute, present)
        };

        // Get a handle to a queue capable of performing graphics commands.
        let graphics_queue = utils::IndexedQueue::get(&logical_device, graphics_index, 0);

        // Get a handle to a queue capable of performing compute commands.
        let compute_queue = utils::IndexedQueue::get(&logical_device, compute_index, 0);

        // Get a handle to a presentation queue for use with swapchain presentation.
        let presentation_queue = utils::IndexedQueue::get(&logical_device, present_index, 0);

        let mut memory_allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: vulkan.instance.clone(),
                device: logical_device.clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: true,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .expect("Unable to create memory allocator (GPU Allocator)");

        // Create an object to manage the swapchain, its images, and synchronization primitives.
        let swapchain = utils::Swapchain::new(
            vulkan,
            physical_device,
            &logical_device,
            surface,
            &mut memory_allocator,
            swapchain_preferences,
            enabled_swapchain_maintenance,
            None,
        );
        let frames_in_flight = swapchain.frames_in_flight();
        let extent = swapchain.extent();
        let image_format = swapchain.image_format();

        // Create a pool for allocating new commands.
        // NOTE: https://developer.nvidia.com/blog/vulkan-dos-donts/ Recommends `image_count * recording_thread_count` many command pools for optimal command buffer allocation.
        //       However, we currently only reuse existing command buffers and do not need to allocate new ones.
        // NOTE: The `RESET_COMMAND_BUFFER` flag allows for resetting individual buffers. If many command buffers are used per frame, setting an entire pool may be more efficient.
        let command_pool = unsafe {
            logical_device
                .create_command_pool(
                    &ash::vk::CommandPoolCreateInfo {
                        flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | ash::vk::CommandPoolCreateFlags::TRANSIENT,
                        queue_family_index: graphics_index,
                        ..Default::default()
                    },
                    None,
                )
                .expect("Unable to create command pool")
        };

        let compute_queue_extra = if graphics_index == compute_index {
            None
        } else {
            #[cfg(debug_assertions)]
            println!("Creating separate compute command pool and compute semaphore");

            let pool = unsafe {
                logical_device.create_command_pool(
                    &ash::vk::CommandPoolCreateInfo {
                        flags: ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | ash::vk::CommandPoolCreateFlags::TRANSIENT,
                        queue_family_index: compute_index,
                        ..Default::default()
                    },
                    None,
                )
            }
            .expect("Unable to create command pool");

            let semaphore = unsafe {
                logical_device
                    .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)
                    .expect("Unable to create semaphore")
            };

            Some((pool, semaphore))
        };

        // Create the FXAA post-processing pass if it is desired.
        // Creating post processing passes first is helpful for chaining passes together.
        let fxaa_pass = if enable_fxaa {
            Some(FxaaPass::new(
                &logical_device,
                &mut memory_allocator,
                extent,
                image_format,
                swapchain.image_views(),
                ash::vk::ImageLayout::PRESENT_SRC_KHR, // The FXAA pass will render directly to the swapchain image for presentation.
            ))
        } else {
            None
        };

        let active_demo = match specialization_constants {
            DemoSpecializationConstants::Triangle(constants) => {
                // Create the graphics pipeline that will be used to render the application.
                let demo = example_triangle::Pipeline::new(
                    &logical_device,
                    None,
                    None,
                    example_triangle::CreateReuseRenderPass::Create {
                        image_format,
                        destination_layout: if enable_fxaa {
                            // NOTE: If we have at least one post-processing pass, we should render to a color attachment.
                            ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        } else {
                            ash::vk::ImageLayout::PRESENT_SRC_KHR
                        },
                    },
                    &swapchain,
                    constants,
                    fxaa_pass.as_ref(),
                );
                DemoPipeline::Triangle(demo)
            }
            DemoSpecializationConstants::Fluid => {
                let demo = example_fluid::FluidSimulation::new(
                    &logical_device,
                    &mut memory_allocator,
                    extent,
                    image_format,
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                    swapchain.image_views(),
                    compute_queue_extra.map_or(command_pool, |(pool, _)| pool),
                    pageable_device_local_memory.as_ref(),
                );
                DemoPipeline::Fluid(demo)
            }
        };

        // Allocate a command buffer for each frame in flight.
        // One may want to use a different number if there are background tasks not related to an image presentation.
        let command_buffer_info = ash::vk::CommandBufferAllocateInfo {
            command_pool,
            level: ash::vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffers = unsafe {
            let mut c = Vec::new();
            c.resize_with(frames_in_flight, || {
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
        let frame_fences = unsafe {
            let mut f = Vec::new();
            f.resize_with(frames_in_flight, || {
                logical_device
                    .create_fence(&fence_create_info, None)
                    .expect("Unable to create fence")
            });
            f
        };

        Self {
            physical_device,
            device_extensions: custom_extensions
                .into_iter()
                .map(|p| unsafe { CStr::from_ptr(p) })
                .collect::<HashSet<_>>(),
            logical_device,
            pageable_device_local_memory,
            surface,
            memory_allocator,

            swapchain,
            active_demo,
            graphics_queue,
            compute_queue,
            presentation_queue,
            command_pool,
            compute_command_pool: compute_queue_extra,

            command_buffers,
            frame_fences,

            fxaa_pass,
            swapchain_preferences,
        }
    }

    /// Destroy the Pompeii renderer and its dependent resources.
    /// # Safety
    /// This function **must** only be called when the owned resources are not currently being processed by the GPU.
    pub fn destroy(mut self, vulkan: &utils::VulkanCore) {
        unsafe {
            // Destroy all fences.
            for fence in self.frame_fences {
                self.logical_device.destroy_fence(fence, None);
            }

            // Destroy all command buffers and the command pool.
            self.logical_device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.logical_device
                .destroy_command_pool(self.command_pool, None);

            // Destroy the graphics pipeline and its dependent resources.
            match self.active_demo {
                DemoPipeline::Triangle(pipeline) => {
                    pipeline.destroy(&self.logical_device, true, true);
                }
                DemoPipeline::Fluid(mut simulation) => {
                    simulation.destroy(&self.logical_device, &mut self.memory_allocator);
                }
            }

            if let Some(mut fxaa_pass) = self.fxaa_pass.take() {
                fxaa_pass.destroy(&self.logical_device, &mut self.memory_allocator);
            }

            // Destroy additional compute resources if the exist.
            if let Some((command_pool, semaphore)) = self.compute_command_pool {
                self.logical_device.destroy_command_pool(command_pool, None);
                self.logical_device.destroy_semaphore(semaphore, None);
            }

            // Destroy the swapchain and its dependent resources.
            self.swapchain
                .destroy(&self.logical_device, &mut self.memory_allocator);

            // Destroy the logical device itself.
            self.logical_device.destroy_device(None);

            // Destroy the Vulkan surface.
            if let Some(khr) = vulkan.khr.as_ref() {
                khr.destroy_surface(self.surface, None);
            } else {
                eprintln!(
                    "ERROR: Unable to destroy surface because the `khr` extension is not available"
                );
            }
        }
    }

    /// Recreate the swapchain, including the framebuffers and image views for the frames owned by the swapchain.
    /// The `self.swapchain_preferences` are used to recreate the swapchain and do not need to match those used with the initial swapchain creation.
    pub fn recreate_swapchain(
        &mut self,
        vulkan: &utils::VulkanCore,
        swapchain_preferences: utils::SwapchainPreferences,
    ) {
        let old_format = self.swapchain.image_format();

        // Recreate the swapchain using the new preferences.
        self.swapchain.recreate_swapchain(
            vulkan,
            self.physical_device,
            &self.logical_device,
            self.surface,
            &mut self.memory_allocator,
            swapchain_preferences,
        );

        // Check if the image format has changed and recreate the render pass and graphics pipeline if necessary.
        let new_swapchain_format = self.swapchain.image_format();
        let extent = self.swapchain.extent();

        // Recreate the FXAA pass if it was enabled.
        if let Some(fxaa_pass) = &mut self.fxaa_pass {
            if new_swapchain_format == old_format {
                fxaa_pass.recreate_framebuffers(
                    &self.logical_device,
                    &mut self.memory_allocator,
                    extent,
                    new_swapchain_format,
                    self.swapchain.image_views(),
                );
            } else {
                let new_fxaa_pass = FxaaPass::new(
                    &self.logical_device,
                    &mut self.memory_allocator,
                    extent,
                    new_swapchain_format,
                    self.swapchain.image_views(),
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                );

                // Destroy the old FXAA pass and replace it with the new one.
                fxaa_pass.destroy(&self.logical_device, &mut self.memory_allocator);

                *fxaa_pass = new_fxaa_pass;
            }
        }

        if new_swapchain_format == old_format {
            // Recreate the framebuffers to account for the new size. Other details are unchanged.
            match &mut self.active_demo {
                DemoPipeline::Triangle(triangle_pipeline) => {
                    triangle_pipeline.recreate_framebuffers(
                        &self.logical_device,
                        &self.swapchain,
                        self.fxaa_pass.as_ref(),
                    );
                }
                DemoPipeline::Fluid(simulation) => {
                    simulation.recreate_framebuffers(
                        &self.logical_device,
                        &mut self.memory_allocator,
                        extent,
                        self.swapchain.image_views(),
                        self.pageable_device_local_memory.as_ref(),
                    );
                }
            }
        } else {
            // Wait for the resources to be available for destruction.
            self.wait_for_tasks();

            // Destroy the old pipeline and recreate the necessary resources.
            match &mut self.active_demo {
                DemoPipeline::Triangle(triangle_pipeline) => {
                    triangle_pipeline.recreate(
                        &self.logical_device,
                        example_triangle::CreateReuseRenderPass::Create {
                            image_format: new_swapchain_format,
                            destination_layout: if self.fxaa_pass.is_some() {
                                ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                            } else {
                                ash::vk::ImageLayout::PRESENT_SRC_KHR
                            },
                        },
                        &self.swapchain,
                        triangle_pipeline.specialization_constants(),
                        self.fxaa_pass.as_ref(),
                    );
                }
                DemoPipeline::Fluid(simulation) => {
                    let new_sim = example_fluid::FluidSimulation::new(
                        &self.logical_device,
                        &mut self.memory_allocator,
                        extent,
                        new_swapchain_format,
                        ash::vk::ImageLayout::PRESENT_SRC_KHR,
                        self.swapchain.image_views(),
                        self.compute_command_pool
                            .map_or(self.command_pool, |(pool, _)| pool),
                        self.pageable_device_local_memory.as_ref(),
                    );
                    simulation.destroy(&self.logical_device, &mut self.memory_allocator);
                    *simulation = new_sim;
                }
            }
        }
    }

    /// Attempt to render the next frame of the application. If there is a recoverable error, then the swapchain is recreated and the function bails early without rendering.
    /// # Panics
    /// * The `utils::VulkanCore` struct must have a `khr` field that is not `None`.
    pub fn render_frame(&mut self, vulkan: &utils::VulkanCore, push_constants: &DemoPushConstants) {
        // Synchronize the CPU with the GPU for the resources previously used for this frame in flight.
        // Specifically, the command buffer cannot be reused until the fence is signaled.
        let current_frame = self.swapchain.current_frame();
        let frame_graphics_fence = self.frame_fences[current_frame];
        let presentation_fence = self.swapchain.present_complete();
        unsafe {
            self.logical_device
                .wait_for_fences(&[frame_graphics_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Unable to wait for fence to begin frame");
        }

        // Get the next image to render to. Has internal synchronization to ensure the previous acquire completed on the GPU.
        let utils::NextSwapchainImage { image_index, .. } = match self
            .swapchain
            .acquire_next_image()
        {
            Ok(f) if !f.suboptimal => f,

            // TODO: Consider accepting suboptimal for this draw, but set a flag to recreate the swapchain next frame.
            Ok(_) | Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let surface_capabilities = unsafe {
                    vulkan
                        .khr
                        .as_ref()
                        .unwrap()
                        .get_physical_device_surface_capabilities(
                            self.physical_device,
                            self.surface,
                        )
                        .expect("Unable to get surface capabilities")
                };
                if surface_capabilities.max_image_extent.width == 0
                    || surface_capabilities.max_image_extent.height == 0
                {
                    println!("WARN: Surface capabilities are zero at image acquire, skipping swapchain recreation");
                    return;
                }

                println!("WARN: Swapchain is out of date at image acquire, needs to be recreated.");
                self.recreate_swapchain(vulkan, self.swapchain_preferences);
                return;
            }

            Err(e) => panic!("Unable to acquire next image from swapchain: {e}"),
        };

        let command_buffer = self.command_buffers[current_frame];
        unsafe {
            // NOTE: We do not need to reset the command buffer here because `ONE_TIME_SUBMIT` command buffers are implicitly reset.
            self.logical_device
                .begin_command_buffer(
                    command_buffer,
                    &ash::vk::CommandBufferBeginInfo {
                        flags: ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .expect("Unable to begin command buffer");
        }

        let extent = self.swapchain.extent();

        // Draw the active demo pipeline.
        match &mut self.active_demo {
            DemoPipeline::Triangle(pipeline) => {
                let DemoPushConstants::Triangle(push_constants) = push_constants else {
                    panic!("Push constants do not match the active demo");
                };
                pipeline.render_frame(
                    &self.logical_device,
                    command_buffer,
                    extent,
                    image_index as usize,
                    push_constants,
                );

                // Add the optional FXAA render pass to the command buffer, if enabled.
                if let Some(fxaa_pass) = &self.fxaa_pass {
                    fxaa_pass.render_frame(
                        &self.logical_device,
                        command_buffer,
                        extent,
                        image_index as usize,
                    );
                }
            }
            DemoPipeline::Fluid(simulation) => {
                let DemoPushConstants::Fluid(push_constants) = push_constants else {
                    panic!("Push constants do not match the active demo");
                };
                simulation.render_frame(
                    &self.logical_device,
                    self.compute_command_pool
                        .map(|(_, compute_semaphore)| compute_semaphore),
                    self.compute_queue.queue,
                    command_buffer,
                    extent,
                    image_index as usize,
                    push_constants,
                    frame_graphics_fence,
                );
            }
        }

        // Complete the graphics command buffer.
        unsafe {
            self.logical_device
                .end_command_buffer(command_buffer)
                .expect("Unable to end the graphics command buffer");
        }

        // Submit the draw command buffer to the GPU.
        let mut semaphores: SmallVec<[_; 2]> = smallvec![self.swapchain.image_available()];
        let mut semaphore_access: SmallVec<[_; 2]> =
            smallvec![ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        if matches!(self.active_demo, DemoPipeline::Fluid(_)) {
            if let Some((_, compute_semaphore)) = &self.compute_command_pool {
                semaphores.push(*compute_semaphore);
                semaphore_access.push(ash::vk::PipelineStageFlags::FRAGMENT_SHADER);
            }
        }
        let submit_info = ash::vk::SubmitInfo {
            wait_semaphore_count: semaphores.len() as u32,
            p_wait_semaphores: semaphores.as_ptr(),
            p_wait_dst_stage_mask: semaphore_access.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 1,
            p_signal_semaphores: &self.swapchain.image_rendered(),
            ..Default::default()
        };
        unsafe {
            if self
                .device_extensions
                .contains(ash::ext::swapchain_maintenance1::NAME)
            {
                self.logical_device
                    .wait_for_fences(&[presentation_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
                    .expect("Unable to wait for fence to end frame");
            }

            self.logical_device
                .reset_fences(&[frame_graphics_fence, presentation_fence])
                .expect("Unable to reset fence for this frame's resources");
            self.logical_device
                .queue_submit(
                    self.graphics_queue.queue,
                    &[submit_info],
                    frame_graphics_fence,
                )
                .expect("Unable to submit command buffer");
        }

        // Queue the presentation of the swapchain image.
        match self.swapchain.present(
            self.presentation_queue.queue,
            self.device_extensions
                .contains(ash::ext::swapchain_maintenance1::NAME),
        ) {
            Ok(_) => (),
            Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let surface_capabilities = unsafe {
                    vulkan
                        .khr
                        .as_ref()
                        .unwrap()
                        .get_physical_device_surface_capabilities(
                            self.physical_device,
                            self.surface,
                        )
                        .expect("Unable to get surface capabilities")
                };
                if surface_capabilities.max_image_extent.width == 0
                    || surface_capabilities.max_image_extent.height == 0
                {
                    #[cfg(debug_assertions)]
                    println!("Surface capabilities are zero at image presentation, skipping swapchain recreation");
                    return;
                }

                println!(
                    "WARN: Swapchain is out of date at image presentation, needs to be recreated."
                );
                self.recreate_swapchain(vulkan, self.swapchain_preferences);
            }
            Err(e) => panic!("Unable to present swapchain image: {e:?}"),
        }
    }

    /// Recreate the graphics pipeline with the new specialization constants.
    /// # Safety
    /// The specialization constants enum type must match the active demo.
    pub fn update_specialization_constants(
        &mut self,
        specialization_constants: DemoSpecializationConstants,
    ) {
        // Wait for the GPU to finish processing all tasks submitted by this renderer.
        self.wait_for_tasks();

        match specialization_constants {
            DemoSpecializationConstants::Triangle(specialization_constants) => {
                let DemoPipeline::Triangle(triangle_pipeline) = &mut self.active_demo else {
                    panic!("Specialization constants do not match the active demo");
                };

                triangle_pipeline.recreate(
                    &self.logical_device,
                    example_triangle::CreateReuseRenderPass::Reuse(triangle_pipeline.render_pass()),
                    &self.swapchain,
                    specialization_constants,
                    self.fxaa_pass.as_ref(),
                );
            }
            DemoSpecializationConstants::Fluid => {}
        }
    }

    /// Wait for the GPU to finish processing all tasks submitted by this renderer.
    fn wait_for_tasks(&self) {
        unsafe {
            if self
                .device_extensions
                .contains(ash::ext::swapchain_maintenance1::NAME)
            {
                let present_fences: SmallVec<[_; EXPECTED_MAX_FRAMES_IN_FLIGHT]> = self
                    .swapchain
                    .frame_syncs()
                    .iter()
                    .map(|f| f.present_complete)
                    .collect();

                self.logical_device
                    .wait_for_fences(&present_fences, true, FIVE_SECONDS_IN_NANOSECONDS)
                    .expect("Unable to wait for present fences to become signaled");
            } else {
                self.logical_device
                    .device_wait_idle()
                    .expect("Unable to wait for device to become idle");
            }
        }
    }

    /// Toggle which demo is currently active.
    pub fn switch_demo(&mut self, new_demo: NewDemo) {
        match new_demo {
            NewDemo::Triangle(constants) => {
                if let DemoPipeline::Triangle(_) = &self.active_demo {
                    return;
                }

                let mut new_triangle = DemoPipeline::Triangle(example_triangle::Pipeline::new(
                    &self.logical_device,
                    None,
                    None,
                    example_triangle::CreateReuseRenderPass::Create {
                        image_format: self.swapchain.image_format(),
                        destination_layout: if self.fxaa_pass.is_some() {
                            ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        } else {
                            ash::vk::ImageLayout::PRESENT_SRC_KHR
                        },
                    },
                    &self.swapchain,
                    constants,
                    self.fxaa_pass.as_ref(),
                ));

                self.wait_for_tasks();
                std::mem::swap(&mut self.active_demo, &mut new_triangle);
                let old_demo = new_triangle; // Rename for clarity.

                match old_demo {
                    DemoPipeline::Fluid(mut simulation) => {
                        simulation.destroy(&self.logical_device, &mut self.memory_allocator);
                    }
                    DemoPipeline::Triangle(_) => {
                        panic!("switch_demo: Wait, I thought we were not using the `Triangle` pipeline at the top of this function...");
                    }
                }
            }
            NewDemo::Fluid => {
                if let DemoPipeline::Fluid(_) = &self.active_demo {
                    return;
                }

                let mut new_fluid = DemoPipeline::Fluid(example_fluid::FluidSimulation::new(
                    &self.logical_device,
                    &mut self.memory_allocator,
                    self.swapchain.extent(),
                    self.swapchain.image_format(),
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                    self.swapchain.image_views(),
                    self.compute_command_pool
                        .map_or(self.command_pool, |(pool, _)| pool),
                    self.pageable_device_local_memory.as_ref(),
                ));

                self.wait_for_tasks();
                std::mem::swap(&mut self.active_demo, &mut new_fluid);
                let old_demo = new_fluid; // Rename for clarity.

                match old_demo {
                    DemoPipeline::Triangle(triangle) => {
                        triangle.destroy(&self.logical_device, true, true);
                    }
                    DemoPipeline::Fluid(_) => {
                        panic!("switch_demo: Wait, I thought we were not using the `Fluid` pipeline at the top of this function...");
                    }
                }
            }
        }
    }
}
