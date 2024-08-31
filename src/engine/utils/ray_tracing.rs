use std::ffi::c_void;

use super::{buffers, EnginePhysicalDeviceFeatures, FIVE_SECONDS_IN_NANOSECONDS};

/// Query the extended properties of a physical device to determine if it supports ray tracing (e.g., RTX).
/// If so, get the ray tracing pipeline properties.
pub fn physical_supports_ray_tracing(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
) -> Option<ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>> {
    // Query the physical device for ray tracing support features.
    let mut features = EnginePhysicalDeviceFeatures::default();
    features.set_ray_tracing(true);
    let _ = features.query_physical_feature_support(instance, physical_device);

    // If the physical device supports ray tracing, return the ray tracing pipeline properties.
    if features.acceleration_structure()
        && features.ray_tracing()
        && features.ray_query()
        && features.ray_tracing_maintenance()
    {
        let mut ray_tracing_pipeline =
            ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut properties =
            ash::vk::PhysicalDeviceProperties2::default().push_next(&mut ray_tracing_pipeline);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };

        // Probably redundant, but ensure no pointers escape this function.
        ray_tracing_pipeline.p_next = std::ptr::null_mut::<c_void>();

        Some(ray_tracing_pipeline)
    } else {
        None
    }
}

/// A temporary buffer necessary in the construction of acceleration structures.
pub struct ScratchBuffer {
    device_address: ash::vk::DeviceAddress,
    buffer_allocation: buffers::BufferAllocation,
    size: u64,
}
impl ScratchBuffer {
    /// Create a new scratch buffer capable of building the desired acceleration structure.
    pub fn new(
        device: &ash::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
        build_scratch_size: u64,
    ) -> Self {
        let buffer_allocation = buffers::new_device_local(
            device,
            pageable_device_local_memory.map(|ext| (ext, 0.4)),
            memory_allocator,
            &ash::vk::BufferCreateInfo::default()
                .size(build_scratch_size)
                .usage(
                    ash::vk::BufferUsageFlags::STORAGE_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
            None,
            "Acceleration Scratch Buffer",
        )
        .expect("Failed to create acceleration scratch buffer");

        let device_address = unsafe {
            device.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(buffer_allocation.buffer),
            )
        };

        Self {
            device_address,
            buffer_allocation,
            size: build_scratch_size,
        }
    }

    /// Destroy the scratch buffer and associated resources.
    pub fn destroy(
        self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        self.buffer_allocation.destroy(device, memory_allocator);
    }

    // Getters.
    pub fn device_address(&self) -> ash::vk::DeviceAddress {
        self.device_address
    }
    pub fn buffer(&self) -> ash::vk::Buffer {
        self.buffer_allocation.buffer
    }
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// A struct ensuring that the `geometries` slice and the `geometry_counts` slice that describes it are the same length.
/// This is necessary for building both top and bottom level acceleration structures.
pub struct InstancedGeometries<'a, 'b> {
    geometries: &'a [ash::vk::AccelerationStructureGeometryKHR<'b>],
    geometry_counts: &'a [u32],
}
#[derive(Debug)]
pub enum InstancedGeometriesError {
    MismatchedSliceLengths,
}
impl<'a, 'b> InstancedGeometries<'a, 'b> {
    /// Create a new list of instanced geometries.
    /// Returns an error if the slices are not the same length.
    pub fn new(
        geometries: &'a [ash::vk::AccelerationStructureGeometryKHR<'b>],
        geometry_counts: &'a [u32],
    ) -> Result<Self, InstancedGeometriesError> {
        if geometries.len() != geometry_counts.len() {
            return Err(InstancedGeometriesError::MismatchedSliceLengths);
        }

        Ok(Self {
            geometries,
            geometry_counts,
        })
    }

    // Getters.
    pub fn geometries(&self) -> &'a [ash::vk::AccelerationStructureGeometryKHR<'b>] {
        self.geometries
    }
    pub fn geometry_counts(&self) -> &'a [u32] {
        self.geometry_counts
    }
    pub fn len(&self) -> usize {
        self.geometries.len()
    }
}

/// A wrapper around a `BufferAllocation` specifically for storing acceleration structure data.
pub struct AccelerationBuffer(buffers::BufferAllocation);
impl AccelerationBuffer {
    /// Create a new acceleration buffer.
    pub fn new(
        device: &ash::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        build_sizes: &ash::vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Self {
        // Create the acceleration structure buffer.
        let acceleration_buffer = buffers::new_device_local(
            device,
            pageable_device_local_memory.map(|d| (d, 1.)),
            allocator,
            &ash::vk::BufferCreateInfo::default()
                .size(build_sizes.acceleration_structure_size)
                .usage(
                    ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
            None,
            "Acceleration Structure Buffer",
        )
        .expect("Failed to create acceleration structure buffer");

        AccelerationBuffer(acceleration_buffer)
    }

    /// Destroy the acceleration buffer and memory.
    pub fn destroy(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        self.0.destroy(device, allocator);
    }

    // Getters.
    pub fn buffer(&self) -> ash::vk::Buffer {
        self.0.buffer
    }
    pub fn allocation(&self) -> &gpu_allocator::vulkan::Allocation {
        &self.0.allocation
    }
}

/// An acceleration structure and its associated memory.
pub struct AccelerationStructure {
    pub handle: ash::vk::AccelerationStructureKHR,
    pub device_address: ash::vk::DeviceAddress,
    pub buffer: AccelerationBuffer,
}
impl AccelerationStructure {
    /// Create a new bottom-level acceleration structure.
    /// # Safety
    /// The `queue` must support compute operations.
    pub fn new_bottom_level(
        device: &ash::Device,
        acceleration_device: &ash::khr::acceleration_structure::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        build_size: &ash::vk::AccelerationStructureBuildSizesInfoKHR,
        scratch_buffer: &ScratchBuffer,
        geometry: InstancedGeometries,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
    ) -> Self {
        // Create the acceleration structure buffer.
        let acceleration_buffer =
            AccelerationBuffer::new(device, pageable_device_local_memory, allocator, build_size);

        // Create the acceleration structure object.
        // NOTE: This does not actually build the underlying acceleration structure.
        //       It just creates the object and framing capable of building for the input geometry.
        let acceleration_structure_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(acceleration_buffer.buffer())
            .offset(acceleration_buffer.allocation().offset())
            .size(build_size.acceleration_structure_size)
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        let acceleration_structure = unsafe {
            acceleration_device.create_acceleration_structure(&acceleration_structure_info, None)
        }
        .expect("Failed to create bottom-level acceleration structure");

        // Get the device address of the acceleration structure.
        // NOTE: This address may be different from `vkGetBufferDeviceAddress` for the underlying buffer.
        //       This value should be used as the source of truth for acceleration structure addresses.
        let device_address = unsafe {
            acceleration_device.get_acceleration_structure_device_address(
                &ash::vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(acceleration_structure),
            )
        };

        let build_geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(acceleration_structure)
            .geometries(geometry.geometries())
            .scratch_data(ash::vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_address(),
            });

        // Describe the range of geometries to build, per geometry type.
        let build_range_infos = geometry
            .geometry_counts()
            .iter()
            .map(|&count| {
                ash::vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)
            })
            .collect::<Vec<_>>();

        // Create a one-time command buffer to build the acceleration structure.
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("Unable to allocate command buffer")[0];
        unsafe {
            device.begin_command_buffer(
                command_buffer,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).expect("Failed to begin command buffer for building bottom-level acceleration structure");
        }

        // Build the acceleration structure.
        unsafe {
            acceleration_device.cmd_build_acceleration_structures(
                command_buffer,
                &[build_geometry_info],
                &[&build_range_infos],
            );
            device.end_command_buffer(command_buffer).expect(
                "Failed to end command buffer for building bottom-level acceleration structure",
            );
        }
        let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None) }
            .expect("Failed to create acceleration structure build fence");
        unsafe {
            device.queue_submit(
                queue,
                &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                fence,
            )
        }
        .expect("Failed to submit acceleration structure build command buffer");

        // Wait for the build to complete.
        // TODO: Allow the caller to wait for the build to complete as needed.
        unsafe { device.wait_for_fences(&[fence], true, FIVE_SECONDS_IN_NANOSECONDS) }
            .expect("Failed to wait for acceleration structure build fence");
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(command_pool, &[command_buffer]);
        };

        Self {
            handle: acceleration_structure,
            device_address,
            buffer: acceleration_buffer,
        }
    }

    /// Create a new top-level acceleration structure.
    /// # Safety
    /// The `queue` must support compute operations.
    /// # Notes
    /// * The function takes a single `AccelerationStructureGeometryKHR` for instance data, and a single `instance_count` describing it
    ///   because the specification requires that the top-level acceleration structure have exactly a single geometry of type `INSTANCES`.
    ///   See <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureBuildGeometryInfoKHR.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03790>.
    pub fn new_top_level(
        device: &ash::Device,
        acceleration_device: &ash::khr::acceleration_structure::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        build_size: &ash::vk::AccelerationStructureBuildSizesInfoKHR,
        scratch_buffer: &ScratchBuffer,
        instance_geometry: &ash::vk::AccelerationStructureGeometryKHR,
        instance_count: u32,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
    ) -> Self {
        // Create the acceleration structure buffer.
        let acceleration_buffer =
            AccelerationBuffer::new(device, pageable_device_local_memory, allocator, build_size);

        // Create the acceleration structure object.
        // NOTE: This does not actually build the underlying acceleration structure.
        //       It just creates the object and framing capable of building for the input geometry.
        let acceleration_structure_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(acceleration_buffer.buffer())
            .offset(acceleration_buffer.allocation().offset())
            .size(build_size.acceleration_structure_size)
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        let acceleration_structure = unsafe {
            acceleration_device.create_acceleration_structure(&acceleration_structure_info, None)
        }
        .expect("Failed to create top-level acceleration structure");

        // Get the device address of the acceleration structure.
        // NOTE: This address may be different from `vkGetBufferDeviceAddress` for the underlying buffer.
        //       This value should be used as the source of truth for acceleration structure addresses.
        let device_address = unsafe {
            acceleration_device.get_acceleration_structure_device_address(
                &ash::vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(acceleration_structure),
            )
        };

        let build_geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(acceleration_structure)
            .geometries(std::slice::from_ref(instance_geometry))
            .scratch_data(ash::vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_address(),
            });

        // Describe the range of geometries to build, per geometry type.
        // NOTE: The `primitive_count` here is the number of instances in the top-level geometry because
        //       that is the primitive from the perspective of the top-level acceleration structure.
        let build_range_info = ash::vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count);

        // Create a one-time command buffer to build the acceleration structure.
        // TODO: Create a utils function or type for single-shot command buffers.
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("Unable to allocate command buffer")[0];
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &ash::vk::CommandBufferBeginInfo::default()
                        .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect(
                    "Failed to begin command buffer for building top-level acceleration structure",
                );
        }

        // Build the acceleration structure.
        unsafe {
            acceleration_device.cmd_build_acceleration_structures(
                command_buffer,
                &[build_geometry_info],
                &[std::slice::from_ref(&build_range_info)],
            );
            device.end_command_buffer(command_buffer).expect(
                "Failed to end command buffer for building top-level acceleration structure",
            );
        }
        let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None) }
            .expect("Failed to create acceleration structure build fence");
        unsafe {
            device.queue_submit(
                queue,
                &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                fence,
            )
        }
        .expect("Failed to submit acceleration structure build command buffer");

        // Wait for the build to complete.
        // TODO: Allow the caller to wait for the build to complete as needed.
        unsafe { device.wait_for_fences(&[fence], true, FIVE_SECONDS_IN_NANOSECONDS) }
            .expect("Failed to wait for acceleration structure build fence");
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(command_pool, &[command_buffer]);
        };

        Self {
            handle: acceleration_structure,
            device_address,
            buffer: acceleration_buffer,
        }
    }

    // Getters.
    pub fn handle(&self) -> ash::vk::AccelerationStructureKHR {
        self.handle
    }
    pub fn device_address(&self) -> ash::vk::DeviceAddress {
        self.device_address
    }

    /// Destroy the acceleration structure and associated resources.
    pub fn destroy(
        self,
        device: &ash::Device,
        acceleration_device: &ash::khr::acceleration_structure::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        unsafe {
            acceleration_device.destroy_acceleration_structure(self.handle, None);
        }
        self.buffer.destroy(device, allocator);
    }
}
