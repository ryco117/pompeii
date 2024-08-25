/// A helper type for managing a buffer and its associated memory allocation.
pub struct BufferAllocation {
    pub buffer: ash::vk::Buffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
}
impl BufferAllocation {
    /// Create a new buffer allocation with the given buffer and allocation.
    pub fn new(buffer: ash::vk::Buffer, allocation: gpu_allocator::vulkan::Allocation) -> Self {
        Self { buffer, allocation }
    }

    /// Destroy the buffer and memory allocation.
    pub fn destroy(
        self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        memory_allocator
            .free(self.allocation)
            .expect("Failed to free buffer memory");
    }
}

/// A helper function for creating a new device-local buffer with the given `VkBufferCreateInfo`.
/// # Safety
/// If a `pageable_device_local_memory` is provided, the `f32` value must be in the range [0, 1] representing a priority with `1` being the greatest.
pub fn new_device_local(
    device: &ash::Device,
    pageable_device_local_memory: Option<(&ash::ext::pageable_device_local_memory::Device, f32)>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    buffer_info: &ash::vk::BufferCreateInfo,
    alignment: Option<u64>,
    name: &str,
) -> Result<BufferAllocation, ash::vk::Result> {
    // Create a new buffer on the device with the desired size and usage flags.
    let (buffer, requirements) = unsafe {
        let buffer = device.create_buffer(buffer_info, None)?;
        let mut requirements = device.get_buffer_memory_requirements(buffer);

        // Optionally, update the alignment requirement if necessary.
        if let Some(alignment) = alignment {
            if requirements.alignment < alignment {
                requirements.alignment = alignment;
            }
        }

        (buffer, requirements)
    };

    // Allocate memory for the buffer in a `GpuOnly` location.
    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true, // "Buffers are always linear" as per README.
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
        })
        .expect("Unable to allocate memory for staging buffer");
    let memory = unsafe { allocation.memory() };

    // Optionally, set the memory priority to allow the driver to optimize the memory usage.
    if let Some((device_ext, priority)) = pageable_device_local_memory {
        unsafe {
            (device_ext.fp().set_device_memory_priority_ext)(device.handle(), memory, priority);
        };
    }

    // Bind the staging buffer to the allocated memory.
    unsafe {
        device.bind_buffer_memory(buffer, memory, allocation.offset())?;
    }

    Ok(BufferAllocation::new(buffer, allocation))
}

/// A wrapper around a `BufferAllocation` that provides a staging buffer suitable for host writes and use as a transfer source to other buffers.
pub struct StagingBuffer(BufferAllocation);

impl StagingBuffer {
    /// Create a new staging buffer with the requested size. The new buffer is suitable for host writes and use as a transfer source to other buffers.
    pub fn new(
        device: &ash::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        size: u64,
    ) -> Result<Self, ash::vk::Result> {
        // Create a new buffer on the device with the necessary size and usage flags.
        let buffer = unsafe {
            device.create_buffer(
                &ash::vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(ash::vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )?
        };

        // Allocate memory for the buffer.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Staging buffer",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true, // "Buffers are always linear" as per README.
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
            })
            .expect("Unable to allocate memory for staging buffer");
        let memory = unsafe { allocation.memory() };

        // Optionally, set the memory priority to allow the driver to optimize the memory usage.
        if let Some(device_ext) = pageable_device_local_memory {
            unsafe {
                (device_ext.fp().set_device_memory_priority_ext)(device.handle(), memory, 0.);
            };
        }

        // Bind the staging buffer to the allocated memory.
        unsafe {
            device.bind_buffer_memory(buffer, memory, allocation.offset())?;
        }

        Ok(Self(BufferAllocation::new(buffer, allocation)))
    }

    /// Destroy the staging buffer and its associated memory allocation.
    pub fn destroy(
        self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        self.0.destroy(device, memory_allocator);
    }

    // Getters.
    pub fn buffer(&self) -> ash::vk::Buffer {
        self.0.buffer
    }
    pub fn allocation(&self) -> &gpu_allocator::vulkan::Allocation {
        &self.0.allocation
    }
    pub fn allocation_mut(&mut self) -> &mut gpu_allocator::vulkan::Allocation {
        &mut self.0.allocation
    }
}

/// Create a new device-local buffer with the given data using a staging buffer.
/// The staging buffer must have sufficient space to hold the data, be writable from the host,
/// and be capable of being used as a transfer source.
/// Returns the new device-local buffer and a fence that can be used to wait for the copy operation to complete.
/// # Note
/// The caller is responsible for ensuring that the fence is waited on and destroyed.
/// # Safety
/// If a `pageable_device_local_memory` is provided, the `f32` value must be in the range [0, 1] representing a priority with `1` being the greatest.
/// # Panics
/// If a staging offset is provided, it must be small enough to provide room to write `data.len()` bytes starting at that offset. Otherwise, the function will panic.
pub fn new_data_buffer(
    device: &ash::Device,
    pageable_device_local_memory: Option<(&ash::ext::pageable_device_local_memory::Device, f32)>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    data: &[u8],
    usage: ash::vk::BufferUsageFlags,
    name: &str,
    staging_buffer: &mut StagingBuffer,
    staging_offset: Option<usize>,
) -> Result<(BufferAllocation, ash::vk::Fence), ash::vk::Result> {
    // Copy the data into the staging buffer.
    let staging_slice = staging_buffer
        .allocation_mut()
        .mapped_slice_mut()
        .expect("Staging buffer did not allocate a mapping");

    // Use the caller provided offset into the staging buffer if one was provided.
    let staging_offset = staging_offset.unwrap_or(0);
    staging_slice[staging_offset..(staging_offset + data.len())].copy_from_slice(data);

    // Create a new device-local buffer.
    let device_buffer = new_device_local(
        device,
        pageable_device_local_memory,
        allocator,
        &ash::vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(usage | ash::vk::BufferUsageFlags::TRANSFER_DST),
        None,
        name,
    )?;

    // Allocate and begin a new command buffer for a one-time copy operation.
    let command_buffer = unsafe {
        device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo {
            command_pool,
            level: ash::vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        })?
    }[0];
    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }

    // Copy the data from the staging buffer to the device-local buffer.
    unsafe {
        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer.buffer(),
            device_buffer.buffer,
            &[ash::vk::BufferCopy {
                src_offset: staging_offset as u64,
                dst_offset: 0,
                size: data.len() as u64,
            }],
        );

        // End the command buffer.
        device.end_command_buffer(command_buffer)?;
    }

    // Create a fence so that the caller can wait for the copy operation to finish.
    let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? };

    // Submit the copy command buffer.
    unsafe {
        device.queue_submit(
            queue,
            &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;
    }

    Ok((device_buffer, fence))
}

/// Update the contents of a device-local buffer with the given data.
/// # Note
/// The caller is responsible for ensuring that the fence is waited on and destroyed.
/// # Safety
/// If a `pageable_device_local_memory` is provided, the `f32` value must be in the range [0, 1] representing a priority with `1` being the greatest.
pub fn update_device_local(
    device: &ash::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    buffer: &BufferAllocation,
    data: &[u8],
    staging_buffer: &mut StagingBuffer,
    staging_offset: Option<usize>,
) -> Result<ash::vk::Fence, ash::vk::Result> {
    // Copy the data into the staging buffer.
    let staging_slice = staging_buffer
        .allocation_mut()
        .mapped_slice_mut()
        .expect("Staging buffer did not allocate a mapping");

    // Use the caller provided offset into the staging buffer if one was provided.
    let staging_offset = staging_offset.unwrap_or(0);
    staging_slice[staging_offset..(staging_offset + data.len())].copy_from_slice(data);

    // Allocate and begin a new command buffer for a one-time copy operation.
    let command_buffer = unsafe {
        device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo {
            command_pool,
            level: ash::vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        })?
    }[0];
    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }

    // Copy the data from the staging buffer to the device-local buffer.
    unsafe {
        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer.buffer(),
            buffer.buffer,
            &[ash::vk::BufferCopy {
                src_offset: staging_offset as u64,
                dst_offset: 0,
                size: data.len() as u64,
            }],
        );

        // End the command buffer.
        device.end_command_buffer(command_buffer)?;
    }

    // Create a fence so that the caller can wait for the copy operation to finish.
    let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? };

    // Submit the copy command buffer.
    unsafe {
        device.queue_submit(
            queue,
            &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;
    }

    Ok(fence)
}
