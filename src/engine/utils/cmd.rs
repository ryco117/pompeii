/// Create a command buffer and write the desired commands to them.
/// The callback `record` can assume that the command buffer parameter is in the recording state and
/// will be ended after the callback returns.
/// # Note
/// The caller is responsible for submitting the command buffer and freeing it.
pub fn record_new_single_shot<F>(
    device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    record: F,
) -> Result<ash::vk::CommandBuffer, ash::vk::Result>
where
    F: FnOnce(ash::vk::CommandBuffer),
{
    // Allocate and begin a new command buffer for a one-time operation.
    let command_buffer = unsafe {
        device.allocate_command_buffers(
            &ash::vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(ash::vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };
    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }

    // Record the desired commands.
    record(command_buffer);

    // End the command buffer.
    unsafe {
        device.end_command_buffer(command_buffer)?;
    }

    Ok(command_buffer)
}

/// Create a command buffer and fence and write the desired commands to them and submit.
/// The callback `record` can assume that the command buffer parameter is in the recording state and
/// will be ended after the callback returns.
/// # Safety
/// The caller must await the fence before reusing or freeing the command buffer.
pub fn submit_single_shot<F>(
    device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    record: F,
) -> Result<(ash::vk::CommandBuffer, ash::vk::Fence), ash::vk::Result>
where
    F: FnOnce(ash::vk::CommandBuffer),
{
    // Record the desired commands to a new command buffer.
    let command_buffer = record_new_single_shot(device, command_pool, record)?;

    // Create a fence so that the caller can wait for the operation to finish.
    let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? };

    // Submit the command buffer.
    unsafe {
        device.queue_submit(
            queue,
            &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;
    }

    Ok((command_buffer, fence))
}
