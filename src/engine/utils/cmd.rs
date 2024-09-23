/// Create a command buffer and fence and write the desired commands to them and submit.
/// The function-value `record` can assume that the command buffer is in the recording state and
/// will be ended after the function returns.
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
    // Allocate and begin a new command buffer for a one-time operation.
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

    // Record the desired commands.
    record(command_buffer);

    // End the command buffer.
    unsafe {
        device.end_command_buffer(command_buffer)?;
    }

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
