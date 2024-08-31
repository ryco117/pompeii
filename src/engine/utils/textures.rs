use super::buffers;

/// Helper type for managing the resources for an allocated image.
//  TODO: Rename to `ImageAllocation`.
pub struct AllocatedImage {
    pub image: ash::vk::Image,
    allocation: gpu_allocator::vulkan::Allocation,
    pub image_view: ash::vk::ImageView,
    pub sampler: Option<ash::vk::Sampler>,
}
impl AllocatedImage {
    /// Create a new image with the given information.
    pub fn new(
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
        image_info: ash::vk::ImageCreateInfo,
        sampler_info: Option<ash::vk::SamplerCreateInfo>,
        image_name: &str,
    ) -> Self {
        let (image, allocation) =
            super::create_image(device, memory_allocator, &image_info, image_name);
        let image_view = super::create_image_view(device, image, image_info.format, 1);

        let sampler = sampler_info.map(|sampler_info| unsafe {
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create the sampler")
        });

        Self {
            image,
            allocation,
            image_view,
            sampler,
        }
    }

    /// Destroy the device image and associated resources.
    pub fn destroy(
        self,
        device: &ash::Device,
        memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        unsafe {
            if let Some(sampler) = self.sampler {
                device.destroy_sampler(sampler, None);
            }
            device.destroy_image_view(self.image_view, None);
            memory_allocator
                .free(self.allocation)
                .expect("Failed to free the image allocation");
            device.destroy_image(self.image, None);
        }
    }
}

/// Copy the given image pixel data into the allocated image.
/// # Safety
/// The image must have a format that is compatible with the data.
pub fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    image: ash::vk::Image,
    extent: ash::vk::Extent2D,
    desired_image_layout: ash::vk::ImageLayout,
    data: &[u8],
    staging_buffer: &mut buffers::StagingBuffer,
    staging_offset: Option<usize>,
) -> Result<super::CleanableFence, ash::vk::Result> {
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

        // Transition the image layout to `TRANSFER_DST_OPTIMAL`.
        let image_barrier = ash::vk::ImageMemoryBarrier2::default()
            .src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
            .src_access_mask(ash::vk::AccessFlags2::NONE)
            .dst_stage_mask(ash::vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(ash::vk::ImageLayout::UNDEFINED)
            .new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        device.cmd_pipeline_barrier2(
            command_buffer,
            &ash::vk::DependencyInfo::default()
                .dependency_flags(ash::vk::DependencyFlags::BY_REGION)
                .image_memory_barriers(&[image_barrier]),
        );

        // Copy the data from the staging buffer to the image.
        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer.buffer(),
            image,
            ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[ash::vk::BufferImageCopy::default()
                .buffer_offset(staging_offset as u64)
                .image_subresource(ash::vk::ImageSubresourceLayers {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(ash::vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })],
        );

        // Transition the image layout to the provided layout.
        let image_barrier = ash::vk::ImageMemoryBarrier2::default()
            .src_stage_mask(ash::vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(ash::vk::PipelineStageFlags2::NONE)
            .dst_access_mask(ash::vk::AccessFlags2::NONE)
            .old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(desired_image_layout)
            .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        device.cmd_pipeline_barrier2(
            command_buffer,
            &ash::vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
        );

        device.end_command_buffer(command_buffer)?;
    }

    // Create a fence so that the caller can wait for the copy operation to finish.
    let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? };
    let cleanable_fence = super::CleanableFence::new(
        fence,
        Some(Box::new(move |device, _| unsafe {
            device.free_command_buffers(command_pool, &[command_buffer]);
        })),
    );

    // Submit the copy command buffer.
    unsafe {
        device.queue_submit(
            queue,
            &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;
    }

    Ok(cleanable_fence)
}
