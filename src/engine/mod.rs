use std::{
    collections::HashSet,
    ffi::CStr,
    sync::{Arc, Mutex},
};

use smallvec::{smallvec, SmallVec};
use utils::{fxaa_pass::FxaaPass, EXPECTED_MAX_FRAMES_IN_FLIGHT, FIVE_SECONDS_IN_NANOSECONDS};

pub mod example_fluid;
pub mod example_ray_tracing;
pub mod example_triangle;
pub mod utils;

/// The demos the application is capable of rendering.
pub enum DemoPipeline {
    Triangle(example_triangle::Pipeline),
    Fluid(Box<example_fluid::FluidSimulation>),
    RayTracing(Box<example_ray_tracing::ExampleRayTracing>),
}

/// The push constants necessary to render the active demo.
/// Each demo needs a unique set of information to render each frame.
pub enum DemoPushConstants {
    Triangle(example_triangle::PushConstants),
    Fluid(example_fluid::ComputePushConstants),
    RayTracing(example_ray_tracing::PushConstants),
}

/// Whether a swapchain resize is necessary or not.
enum ResizeSwapchainState {
    None,
    Resized,
}

/// The expected maximum number of additional fences that may need to be waited on before the next frame can be rendered.
const EXPECTED_MAX_FENCES_IN_FLIGHT: usize = 2;

/// Define which rendering objects are necessary for this application.
pub struct Renderer {
    physical_device: ash::vk::PhysicalDevice,
    device_extensions: HashSet<&'static CStr>,
    pub logical_device: ash::Device,
    pageable_device_local_memory: Option<ash::ext::pageable_device_local_memory::Device>,
    ray_tracing: Option<(
        ash::khr::acceleration_structure::Device,
        ash::khr::ray_tracing_pipeline::Device,
        ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    )>,

    // Manage GPU memory with an efficient allocation strategy.
    memory_allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,

    surface: ash::vk::SurfaceKHR,
    swapchain: utils::Swapchain,
    resize_swapchain: ResizeSwapchainState,

    /// The specific object we are interested in rendering.
    pub active_demo: DemoPipeline,

    graphics_queue: utils::IndexedQueue,
    compute_queue: utils::IndexedQueue,
    presentation_queue: utils::IndexedQueue,
    command_pool: ash::vk::CommandPool,
    compute_command_pool: Option<(ash::vk::CommandPool, ash::vk::Semaphore)>, // Optional compute command pool and compute semaphore if the graphics and compute queue families are separate.

    command_buffers: Vec<ash::vk::CommandBuffer>,

    /// The graphics queue submit fence for each frame in flight.
    frame_fences: Vec<ash::vk::Fence>,

    /// Fences that must be waited on before the next frame can be rendered.
    /// Fences will be added and removed as dependencies are created and resolved.
    blocking_fences: SmallVec<[utils::CleanableFence; EXPECTED_MAX_FENCES_IN_FLIGHT]>,

    fxaa_pass: Option<FxaaPass>,
    pub swapchain_preferences: utils::SwapchainPreferences,
}

#[derive(Clone, Copy, Debug)]
pub enum DemoSpecializationConstants {
    Triangle(example_triangle::SpecializationConstants),

    Fluid,
}

/// The new demo to switch to, and any unique parameters necessary to initialize it.
#[derive(Clone, Copy, Debug)]
pub enum NewDemo {
    Triangle(example_triangle::SpecializationConstants),
    Fluid,
    RayTracing,
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
        let mut custom_extensions = Vec::new();
        let available_extensions = unsafe {
            vulkan
                .instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Unable to enumerate device extensions")
        };
        let enabled_swapchain_maintenance =
            if vulkan.enabled_instance_extension(ash::ext::surface_maintenance1::NAME) {
                utils::extend_if_all_extensions_are_contained(
                    &available_extensions,
                    &[ash::ext::swapchain_maintenance1::NAME],
                    &mut custom_extensions,
                    #[cfg(debug_assertions)]
                    utils::ExtensionType::Device,
                )
            } else {
                false
            };
        let enabled_ray_tracing_device_extension = utils::extend_if_all_extensions_are_contained(
            &available_extensions,
            &[
                ash::khr::deferred_host_operations::NAME,
                ash::khr::acceleration_structure::NAME,
                ash::khr::ray_query::NAME,
                ash::khr::ray_tracing_pipeline::NAME,
                ash::khr::ray_tracing_maintenance1::NAME,
            ],
            &mut custom_extensions,
            #[cfg(debug_assertions)]
            utils::ExtensionType::Device,
        );

        // Determine if the physical device supports ray tracing.
        let mut ray_tracing = if enabled_ray_tracing_device_extension {
            utils::ray_tracing::physical_supports_ray_tracing(&vulkan.instance, physical_device)
        } else {
            None
        };

        let enabled_pageable_device_local_memory = utils::extend_if_all_extensions_are_contained(
            &available_extensions,
            &[
                ash::ext::memory_priority::NAME,
                ash::ext::pageable_device_local_memory::NAME,
            ],
            &mut custom_extensions,
            #[cfg(debug_assertions)]
            utils::ExtensionType::Device,
        ) && device_features
            .pageable_device_local_memory();
        if ray_tracing.is_some() {
            // Ensure that we toggle bits to vocalize our intent to enable ray tracing.
            device_features.set_ray_tracing(true);
        }

        // We required these features in physical device selection.
        // Now we can request the features be enabled over the logical device we create.
        // Ignore the features we are not interested in, and enable the ones we are.
        let mut features = ash::vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut device_features.dynamic_rendering)
            .push_next(&mut device_features.synchronization2)
            .push_next(&mut device_features.buffer_device_address);

        // Add optional device features when they are available.
        if enabled_pageable_device_local_memory {
            #[cfg(debug_assertions)]
            println!("INFO: Enabling VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT");
            features = features.push_next(&mut device_features.pageable_device_local_memory);
        }
        if device_features
            .descriptor_indexing
            .shader_sampled_image_array_non_uniform_indexing
            == ash::vk::TRUE
            && device_features.descriptor_indexing.runtime_descriptor_array == ash::vk::TRUE
        {
            #[cfg(debug_assertions)]
            println!("INFO: Enabling VkPhysicalDeviceDescriptorIndexingFeatures");
            features = features.push_next(&mut device_features.descriptor_indexing);
        } else {
            eprintln!("ERROR: Unable to enable descriptor indexing features");

            // Disable ray tracing if descriptor indexing is not available.
            ray_tracing = None;
        }
        if ray_tracing.is_some() {
            #[cfg(debug_assertions)]
            println!("INFO: Enabling VkPhysicalDeviceAccelerationStructureFeaturesKHR");
            features = features.push_next(&mut device_features.acceleration_structure);

            #[cfg(debug_assertions)]
            println!("INFO: Enabling VkPhysicalDeviceRayQueryFeaturesKHR");
            features = features.push_next(&mut device_features.ray_query);

            #[cfg(debug_assertions)]
            println!("INFO: Enabling VkPhysicalDeviceRayTracingPipelineFeaturesKHR");
            features = features.push_next(&mut device_features.ray_tracing);

            #[cfg(debug_assertions)]
            println!("INFO: Enabling VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR");
            features = features.push_next(&mut device_features.ray_tracing_maintenance);
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

        // Create device instances for the optional features enabled on this device.
        let pageable_device_local_memory = if enabled_pageable_device_local_memory {
            Some(ash::ext::pageable_device_local_memory::Device::new(
                &vulkan.instance,
                &logical_device,
            ))
        } else {
            None
        };
        let ray_tracing = ray_tracing.map(|p| {
            #[cfg(debug_assertions)]
            println!("INFO: Enabling ray tracing pipeline with properties: {p:?}");

            (
                ash::khr::acceleration_structure::Device::new(&vulkan.instance, &logical_device),
                ash::khr::ray_tracing_pipeline::Device::new(&vulkan.instance, &logical_device),
                p,
            )
        });

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
                    .expect("Unable to find a queue family supporting presentation")
            };

            // NOTE: Prefer that the graphics and compute queues are equivalent because the `example_fluid` module will benefit from shared resources.
            let compute = if queue_families.compute.contains(&graphics) {
                graphics
            } else {
                *queue_families
                    .compute
                    .first()
                    .expect("Unable to find a queue family supporting compute")
            };

            (graphics, compute, present)
        };

        // Get a handle to a queue capable of performing graphics commands.
        let graphics_queue = utils::IndexedQueue::get(&logical_device, graphics_index, 0);

        // Get a handle to a queue capable of performing compute commands.
        let compute_queue = utils::IndexedQueue::get(&logical_device, compute_index, 0);

        // Get a handle to a presentation queue for use with swapchain presentation.
        let presentation_queue = utils::IndexedQueue::get(&logical_device, present_index, 0);

        // Use a memory allocator to manage GPU memory with an efficient allocation strategy.
        let memory_allocator = Mutex::new(
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: vulkan.instance.clone(),
                device: logical_device.clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: true,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .expect("Unable to create memory allocator (GPU Allocator)"),
        );

        // Create an object to manage the swapchain, its images, and synchronization primitives.
        let swapchain = utils::Swapchain::new(
            vulkan,
            physical_device,
            &logical_device,
            surface,
            &memory_allocator,
            Some(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT | ash::vk::ImageUsageFlags::STORAGE),
            swapchain_preferences,
            enabled_swapchain_maintenance,
            None,
        );
        let frames_in_flight = swapchain.frames_in_flight();
        let extent = swapchain.extent();
        let image_format = swapchain.image_format();

        // Create a pool for allocating new commands.
        // NOTE: https://developer.nvidia.com/blog/vulkan-dos-donts/ Recommends `frame_count * recording_thread_count` many command pools for optimal command buffer allocation.
        //       However, we currently only reuse existing command buffers and do not need to allocate new ones.
        // NOTE: The `RESET_COMMAND_BUFFER` flag allows for resetting individual buffers. If many command buffers are used per frame, setting an entire pool may be more efficient.
        let command_pool = unsafe {
            logical_device
                .create_command_pool(
                    &ash::vk::CommandPoolCreateInfo::default()
                        .flags(
                            ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                                | ash::vk::CommandPoolCreateFlags::TRANSIENT,
                        )
                        .queue_family_index(graphics_index),
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
                    &ash::vk::CommandPoolCreateInfo::default()
                        .flags(
                            ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                                | ash::vk::CommandPoolCreateFlags::TRANSIENT,
                        )
                        .queue_family_index(compute_index),
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
                &memory_allocator,
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
                    &memory_allocator,
                    extent,
                    image_format,
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                    swapchain.image_views(),
                    compute_queue_extra.map_or(command_pool, |(pool, _)| pool),
                    compute_queue.queue,
                );
                DemoPipeline::Fluid(Box::new(demo))
            }
        };

        // Allocate a command buffer for each frame in flight.
        // One may want to use a different number if there are background tasks not related to an image presentation.
        let command_buffer_info = ash::vk::CommandBufferAllocateInfo {
            command_pool,
            level: ash::vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: frames_in_flight as u32,
            ..Default::default()
        };
        let command_buffers = unsafe {
            logical_device
                .allocate_command_buffers(&command_buffer_info)
                .expect("Unable to allocate command buffer")
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
            ray_tracing,

            memory_allocator: Arc::new(memory_allocator),

            surface,
            swapchain,
            resize_swapchain: ResizeSwapchainState::None,

            active_demo,
            graphics_queue,
            compute_queue,
            presentation_queue,
            command_pool,
            compute_command_pool: compute_queue_extra,

            command_buffers,
            frame_fences,
            blocking_fences: SmallVec::new(),

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
            // NOTE: Destroying the pool frees all command buffers allocated from it.
            self.logical_device
                .destroy_command_pool(self.command_pool, None);

            // Destroy the graphics pipeline and its dependent resources.
            match self.active_demo {
                DemoPipeline::Triangle(pipeline) => {
                    pipeline.destroy(&self.logical_device, true, true);
                }
                DemoPipeline::Fluid(mut simulation) => {
                    simulation.destroy(&self.logical_device, &self.memory_allocator);
                }
                DemoPipeline::RayTracing(ray_tracing) => {
                    ray_tracing.destroy(
                        &self.logical_device,
                        &self.ray_tracing.as_ref().unwrap().0,
                        &self.memory_allocator,
                    );
                }
            }

            if let Some(mut fxaa_pass) = self.fxaa_pass.take() {
                fxaa_pass.destroy(&self.logical_device, &self.memory_allocator);
            }

            // Destroy additional compute resources if the exist.
            if let Some((command_pool, semaphore)) = self.compute_command_pool {
                self.logical_device.destroy_command_pool(command_pool, None);
                self.logical_device.destroy_semaphore(semaphore, None);
            }

            // Destroy the swapchain and its dependent resources.
            self.swapchain
                .destroy(&self.logical_device, &self.memory_allocator);

            // Destroy the logical device itself.
            self.logical_device.destroy_device(None);

            // Destroy the Vulkan surface.
            if let Some(khr) = vulkan.surface_instance.as_ref() {
                khr.destroy_surface(self.surface, None);
            } else {
                eprintln!(
                    "ERROR: Unable to destroy surface because the `khr` extension is not available"
                );
            }
        }
    }

    /// Get the extent of the swapchain images.
    pub fn swapchain_extent(&self) -> ash::vk::Extent2D {
        self.swapchain.extent()
    }

    /// Indicate that the swapchain needs to be recreated before next use.
    /// Generally, this is used when the window is resized. However, other changes may require a swapchain recreation.
    pub fn swapchain_recreation_required(&mut self, new_extent: Option<ash::vk::Extent2D>) {
        if let Some(extent) = new_extent {
            self.swapchain_preferences.preferred_extent = Some(extent);
        }
        self.resize_swapchain = ResizeSwapchainState::Resized;
    }

    /// Handle any impending swapchain recreations.
    pub fn handle_swapchain_resize(&mut self, vulkan: &utils::VulkanCore) {
        if matches!(self.resize_swapchain, ResizeSwapchainState::Resized) {
            self.recreate_swapchain(vulkan);
        }
    }

    /// Recreate the swapchain, including the framebuffers and image views for the frames owned by the swapchain.
    /// The `self.swapchain_preferences` are used to recreate the swapchain and do not need to match those used with the initial swapchain creation.
    pub fn recreate_swapchain(&mut self, vulkan: &utils::VulkanCore) {
        let old_format = self.swapchain.image_format();

        // Recreate the swapchain using the new preferences.
        self.swapchain.recreate_swapchain(
            vulkan,
            self.physical_device,
            &self.logical_device,
            self.surface,
            &self.memory_allocator,
            Some(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT | ash::vk::ImageUsageFlags::STORAGE),
            self.swapchain_preferences,
        );

        // Check if the image format has changed and recreate the render pass and graphics pipeline if necessary.
        let new_swapchain_format = self.swapchain.image_format();
        let extent = self.swapchain.extent();

        // Recreate the FXAA pass if it was enabled.
        if let Some(fxaa_pass) = &mut self.fxaa_pass {
            if new_swapchain_format == old_format {
                fxaa_pass.recreate_framebuffers(
                    &self.logical_device,
                    &self.memory_allocator,
                    extent,
                    new_swapchain_format,
                    self.swapchain.image_views(),
                );
            } else {
                let new_fxaa_pass = FxaaPass::new(
                    &self.logical_device,
                    &self.memory_allocator,
                    extent,
                    new_swapchain_format,
                    self.swapchain.image_views(),
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                );

                // Destroy the old FXAA pass and replace it with the new one.
                fxaa_pass.destroy(&self.logical_device, &self.memory_allocator);

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
                        &self.memory_allocator,
                        self.compute_command_pool
                            .map_or(self.command_pool, |(pool, _)| pool),
                        self.compute_queue.queue,
                        extent,
                        self.swapchain.image_views(),
                    );
                }
                DemoPipeline::RayTracing(tracer) => self.blocking_fences.push(
                    tracer.recreate_view_sets(
                        &self.logical_device,
                        self.pageable_device_local_memory.as_ref(),
                        &self.memory_allocator,
                        self.compute_command_pool
                            .map_or(self.command_pool, |(pool, _)| pool),
                        self.compute_queue.queue,
                        self.swapchain.image_views(),
                        extent,
                    ),
                ),
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
                        &self.memory_allocator,
                        extent,
                        new_swapchain_format,
                        ash::vk::ImageLayout::PRESENT_SRC_KHR,
                        self.swapchain.image_views(),
                        self.compute_command_pool
                            .map_or(self.command_pool, |(pool, _)| pool),
                        self.compute_queue.queue,
                    );
                    simulation.destroy(&self.logical_device, &self.memory_allocator);
                    *simulation = Box::new(new_sim);
                }
                DemoPipeline::RayTracing(tracer) => self.blocking_fences.push(
                    tracer.recreate_view_sets(
                        &self.logical_device,
                        self.pageable_device_local_memory.as_ref(),
                        &self.memory_allocator,
                        self.compute_command_pool
                            .map_or(self.command_pool, |(pool, _)| pool),
                        self.compute_queue.queue,
                        self.swapchain.image_views(),
                        extent,
                    ),
                ),
            }
        }

        // Reset the flag indicating the swapchain needs to be recreated.
        self.resize_swapchain = ResizeSwapchainState::None;
    }

    /// Attempt to render the next frame of the application. If there is a recoverable error, then the swapchain is recreated and the function bails early without rendering.
    /// # Panics
    /// * The `utils::VulkanCore` struct must have a `khr` field that is not `None`.
    pub fn render_frame(&mut self, vulkan: &utils::VulkanCore, push_constants: &DemoPushConstants) {
        // Synchronize the CPU with the GPU for the resources previously used for this frame index.
        // Specifically, the command buffer cannot be reused until the fence is signaled.
        let current_frame = self.swapchain.current_frame();
        let frame_graphics_fence = self.frame_fences[current_frame];
        let presentation_fence = self.swapchain.present_complete();
        unsafe {
            let (blocking_fences, blocking_fence_cleanup): (Vec<_>, Vec<_>) = self
                .blocking_fences
                .drain(..)
                .map(|utils::CleanableFence { fence, cleanup }| (fence, cleanup))
                .unzip();
            let blocking_fences: SmallVec<[_; EXPECTED_MAX_FENCES_IN_FLIGHT + 1]> =
                std::iter::once(frame_graphics_fence)
                    .chain(blocking_fences)
                    .collect();
            self.logical_device
                .wait_for_fences(&blocking_fences, true, FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Unable to wait for fence to begin frame");

            // Destroy the one-time fences and any resources they have.
            for fence in &blocking_fences[1..] {
                self.logical_device.destroy_fence(*fence, None);
            }

            for cleanup in blocking_fence_cleanup.into_iter().flatten() {
                cleanup(&self.logical_device, &self.memory_allocator);
            }
        }

        // Get the next image to render to. Has internal synchronization to ensure the previous acquire completed on the GPU.
        let utils::NextSwapchainImage {
            image_index,
            suboptimal,
        } = match self.swapchain.acquire_next_image() {
            Ok(f) => f,

            Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let surface_capabilities = unsafe {
                    vulkan
                        .surface_instance
                        .as_ref()
                        .unwrap()
                        .get_physical_device_surface_capabilities(
                            self.physical_device,
                            self.surface,
                        )
                        .expect("Unable to get surface capabilities")
                };

                // Zero-sized surfaces are a special case where we should not recreate or render
                // until a non-zero size is requested.
                if surface_capabilities.max_image_extent.width == 0
                    || surface_capabilities.max_image_extent.height == 0
                {
                    println!("WARN: Surface capabilities are zero at image acquire, skipping swapchain recreation");
                    return;
                }

                println!("WARN: Swapchain is out of date at image acquire, needs to be recreated");
                self.swapchain_recreation_required(
                    if surface_capabilities.current_extent == utils::SPECIAL_SURFACE_EXTENT {
                        None
                    } else {
                        Some(surface_capabilities.current_extent)
                    },
                );
                return;
            }

            Err(e) => panic!("Unable to acquire next image from swapchain: {e}"),
        };

        // Ensure suboptimal images are acknowledged.
        if suboptimal {
            println!(
                "WARN: Swapchain image is suboptimal, recreating the swapchain after this frame"
            );
            self.resize_swapchain = ResizeSwapchainState::None;
        }

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
            DemoPipeline::RayTracing(ray_tracing) => {
                let DemoPushConstants::RayTracing(push_constants) = push_constants else {
                    panic!("Push constants do not match the active demo");
                };
                ray_tracing.record_command_buffer(
                    &self.logical_device,
                    &self.ray_tracing.as_ref().unwrap().1,
                    command_buffer,
                    &self.ray_tracing.as_ref().unwrap().2,
                    self.swapchain.images()[image_index as usize],
                    image_index,
                    extent,
                    push_constants,
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
                        .surface_instance
                        .as_ref()
                        .unwrap()
                        .get_physical_device_surface_capabilities(
                            self.physical_device,
                            self.surface,
                        )
                        .expect("Unable to get surface capabilities")
                };

                // Zero-sized surfaces are a special case where we should not recreate or render
                // until a non-zero size is requested.
                if surface_capabilities.max_image_extent.width == 0
                    || surface_capabilities.max_image_extent.height == 0
                {
                    #[cfg(debug_assertions)]
                    println!("Surface capabilities are zero at image presentation, skipping swapchain recreation");
                    return;
                }

                println!(
                    "WARN: Swapchain is out of date at image presentation, needs to be recreated"
                );
                self.swapchain_recreation_required(
                    if surface_capabilities.current_extent == utils::SPECIAL_SURFACE_EXTENT {
                        None
                    } else {
                        Some(surface_capabilities.current_extent)
                    },
                );
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
        let mut new_demo = match new_demo {
            NewDemo::Triangle(constants) => {
                if let DemoPipeline::Triangle(_) = &self.active_demo {
                    return;
                }

                DemoPipeline::Triangle(example_triangle::Pipeline::new(
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
                ))
            }
            NewDemo::Fluid => {
                if let DemoPipeline::Fluid(_) = &self.active_demo {
                    return;
                }

                DemoPipeline::Fluid(Box::new(example_fluid::FluidSimulation::new(
                    &self.logical_device,
                    &self.memory_allocator,
                    self.swapchain.extent(),
                    self.swapchain.image_format(),
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                    self.swapchain.image_views(),
                    self.compute_command_pool
                        .map_or(self.command_pool, |(pool, _)| pool),
                    self.compute_queue.queue,
                )))
            }
            NewDemo::RayTracing => {
                if let DemoPipeline::RayTracing(_) = &self.active_demo {
                    return;
                }

                let (command_pool, queue) = self
                    .compute_command_pool
                    .map_or((self.command_pool, &self.graphics_queue), |(pool, _)| {
                        (pool, &self.compute_queue)
                    });
                DemoPipeline::RayTracing(Box::new(example_ray_tracing::ExampleRayTracing::new(
                    &self.logical_device,
                    &self.ray_tracing.as_ref().unwrap().0,
                    &self.ray_tracing.as_ref().unwrap().1,
                    self.pageable_device_local_memory.as_ref(),
                    &self.memory_allocator,
                    command_pool,
                    queue,
                    self.swapchain.image_views(),
                    self.swapchain.extent(),
                    &self.ray_tracing.as_ref().unwrap().2,
                )))
            }
        };

        self.wait_for_tasks();
        std::mem::swap(&mut self.active_demo, &mut new_demo);
        let old_demo = new_demo; // Rename for clarity after swapping.

        // Delete the old demo.
        match old_demo {
            DemoPipeline::Triangle(triangle) => {
                triangle.destroy(&self.logical_device, true, true);
            }
            DemoPipeline::Fluid(mut simulation) => {
                simulation.destroy(&self.logical_device, &self.memory_allocator);
            }
            DemoPipeline::RayTracing(ray_tracing) => {
                ray_tracing.destroy(
                    &self.logical_device,
                    &self.ray_tracing.as_ref().unwrap().0,
                    &self.memory_allocator,
                );
            }
        }
    }

    /// Check if the renderer is capable of using the Vulkan ray tracing pipeline.
    pub fn ray_tracing_enabled(&self) -> bool {
        self.ray_tracing.is_some()
    }
}
