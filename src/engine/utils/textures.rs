use std::sync::Mutex;

use super::buffers;

/// Minimal helper type for consistent pixel representations on textures.
#[repr(C)]
pub struct TexturePixel {
    p: [u8; 4],
}
impl TexturePixel {
    pub fn new(p: [u8; 4]) -> Self {
        Self { p }
    }
}

/// Determine the number of bytes in each glTF texture pixel given the input format.
pub fn pixel_format_length(format: gltf::image::Format) -> usize {
    use gltf::image::Format;
    match format {
        // 8 bits per channel.
        Format::R8 => 1,
        Format::R8G8 => 2,
        Format::R8G8B8 => 3,
        Format::R8G8B8A8 => 4,

        // 16 bits per channel.
        Format::R16 => 2,
        Format::R16G16 => 4,
        Format::R16G16B16 => 6,
        Format::R16G16B16A16 => 8,

        // 32 bits per channel.
        Format::R32G32B32FLOAT => 12,
        Format::R32G32B32A32FLOAT => 16,
    }
}

/// Convert a byte-slice to a pixel based on the glTF format.
pub fn bytes_to_pixel(bytes: &[u8], source_format: gltf::image::Format) -> TexturePixel {
    use gltf::image::Format;

    #[cfg(debug_assertions)]
    assert_eq!(
        bytes.len(),
        pixel_format_length(source_format),
        "Invalid number of bytes for pixel format"
    );

    let mut pixel = [0; 4];
    match source_format {
        // 8 bits per channel.
        Format::R8 => {
            pixel[0] = bytes[0];
        }
        Format::R8G8 => {
            pixel[0] = bytes[0];
            pixel[1] = bytes[1];
        }
        Format::R8G8B8 => {
            pixel[0] = bytes[0];
            pixel[1] = bytes[1];
            pixel[2] = bytes[2];
        }
        Format::R8G8B8A8 => {
            pixel[0] = bytes[0];
            pixel[1] = bytes[1];
            pixel[2] = bytes[2];
            pixel[3] = bytes[3];
        }

        // 16 bits per channel.
        Format::R16 => {
            // Ignore the low order bits.
            pixel[0] = bytes[1];
        }
        Format::R16G16 => {
            // Ignore the low order bits.
            pixel[0] = bytes[1];
            pixel[1] = bytes[3];
        }
        Format::R16G16B16 => {
            // Ignore the low order bits.
            pixel[0] = bytes[1];
            pixel[1] = bytes[3];
            pixel[2] = bytes[5];
        }
        Format::R16G16B16A16 => {
            // Ignore the low order bits.
            pixel[0] = bytes[1];
            pixel[1] = bytes[3];
            pixel[2] = bytes[5];
            pixel[3] = bytes[7];
        }

        // 32 bits per channel.
        Format::R32G32B32FLOAT => {
            pixel[0] = (255. * f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])) as u8;
            pixel[1] = (255. * f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])) as u8;
            pixel[2] =
                (255. * f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]])) as u8;
        }
        Format::R32G32B32A32FLOAT => {
            pixel[0] = (255. * f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])) as u8;
            pixel[1] = (255. * f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])) as u8;
            pixel[2] =
                (255. * f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]])) as u8;
            pixel[3] =
                (255. * f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]])) as u8;
        }
    }
    TexturePixel::new(pixel)
}

/// Helper type for managing the resources for an allocated image.
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
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
        image_info: ash::vk::ImageCreateInfo,
        sampler_info: Option<ash::vk::SamplerCreateInfo>,
        image_name: &str,
    ) -> Self {
        let (image, allocation) =
            { super::create_image(device, memory_allocator, &image_info, image_name) };
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
        memory_allocator: &Mutex<gpu_allocator::vulkan::Allocator>,
    ) {
        unsafe {
            if let Some(sampler) = self.sampler {
                device.destroy_sampler(sampler, None);
            }
            device.destroy_image_view(self.image_view, None);
            memory_allocator
                .lock()
                .expect("Failed to lock allocator for `AllocatedImage::destroy`")
                .free(self.allocation)
                .expect("Failed to free the image allocation");
            device.destroy_image(self.image, None);
        }
    }
}

/// Create a new one-time command buffer to copy the given image pixel data into the allocated image.
/// # Safety
/// The image must have a format that is compatible with the data.
pub fn copy_buffer_to_image_command(
    device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    image: ash::vk::Image,
    extent: ash::vk::Extent2D,
    desired_image_layout: ash::vk::ImageLayout,
    data: &[u8],
    staging_buffer: &mut buffers::StagingBuffer,
    staging_offset: Option<usize>,
) -> Result<ash::vk::CommandBuffer, ash::vk::Result> {
    // Copy the data into the staging buffer.
    let staging_slice = staging_buffer
        .allocation_mut()
        .mapped_slice_mut()
        .expect("Staging buffer did not allocate a mapping");

    // Use the caller provided offset into the staging buffer if one was provided.
    let staging_offset = staging_offset.unwrap_or(0);
    staging_slice[staging_offset..(staging_offset + data.len())].copy_from_slice(data);

    // Record the copy commands to a new command buffer.
    let command_buffer =
        super::cmd::record_new_single_shot(device, command_pool, |command_buffer| {
            unsafe {
                // Transition the image layout to `TRANSFER_DST_OPTIMAL`.
                record_image_layout_transition(
                    device,
                    command_buffer,
                    image,
                    ash::vk::ImageLayout::UNDEFINED,
                    ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ash::vk::PipelineStageFlags2::NONE,
                    ash::vk::AccessFlags2::NONE,
                    ash::vk::PipelineStageFlags2::TRANSFER,
                    ash::vk::AccessFlags2::TRANSFER_WRITE,
                    ash::vk::DependencyFlags::BY_REGION,
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
                record_image_layout_transition(
                    device,
                    command_buffer,
                    image,
                    ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    desired_image_layout,
                    ash::vk::PipelineStageFlags2::TRANSFER,
                    ash::vk::AccessFlags2::TRANSFER_WRITE,
                    ash::vk::PipelineStageFlags2::NONE,
                    ash::vk::AccessFlags2::NONE,
                    ash::vk::DependencyFlags::empty(),
                );
            }
        })?;

    Ok(command_buffer)
}

/// Create a new one-time command buffer to copy the given image pixel data into the allocated image.
/// # Safety
/// The image must have a format that is compatible with the data.
pub fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    queue: &Mutex<ash::vk::Queue>,
    image: ash::vk::Image,
    extent: ash::vk::Extent2D,
    desired_image_layout: ash::vk::ImageLayout,
    data: &[u8],
    staging_buffer: &mut buffers::StagingBuffer,
    staging_offset: Option<usize>,
) -> Result<super::CleanableFence, ash::vk::Result> {
    // Record the copy commands to a new command buffer.
    let command_buffer = copy_buffer_to_image_command(
        device,
        command_pool,
        image,
        extent,
        desired_image_layout,
        data,
        staging_buffer,
        staging_offset,
    )?;

    // Create a fence so that the caller can wait for the operation to finish.
    let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? };

    // Submit the command buffer.
    {
        let queue_lock = queue
            .lock()
            .expect("Failed to lock the `copy_buffer_to_image` queue");
        unsafe {
            device.queue_submit(
                *queue_lock,
                &[ash::vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                fence,
            )?;
        }
    }

    // Create a `CleanableFence` so that the caller can free the command buffer when available.
    let cleanable_fence = super::CleanableFence::new(
        fence,
        Some(Box::new(move |device, _| unsafe {
            device.free_command_buffers(command_pool, &[command_buffer]);
        })),
    );

    Ok(cleanable_fence)
}

/// Helper to record an image layout transition in a command buffer.
pub fn record_image_layout_transition(
    device: &ash::Device,
    command_buffer: ash::vk::CommandBuffer,
    image: ash::vk::Image,
    old_layout: ash::vk::ImageLayout,
    new_layout: ash::vk::ImageLayout,
    src_stage_mask: ash::vk::PipelineStageFlags2,
    src_access_mask: ash::vk::AccessFlags2,
    dst_stage_mask: ash::vk::PipelineStageFlags2,
    dst_access_mask: ash::vk::AccessFlags2,
    dependency_flags: ash::vk::DependencyFlags,
) {
    let image_barrier = ash::vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
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
    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &ash::vk::DependencyInfo::default()
                .dependency_flags(dependency_flags)
                .image_memory_barriers(&[image_barrier]),
        );
    }
}
