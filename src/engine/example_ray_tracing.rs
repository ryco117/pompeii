use crate::engine::utils::{
    self, create_shader_module, shaders::ENTRY_POINT_MAIN, FIVE_SECONDS_IN_NANOSECONDS,
};

use nalgebra_glm as glm;

/// The maximum number of traversal iterations to allow before a ray must terminate.
const MAX_RAY_RECURSION_DEPTH: u32 = 10;

/// The maximum number of textures that can be used in the ray tracing pipeline.
const MAX_TEXTURES: u32 = 1024;

/// Special index to indicate that a resource is not used.
const MISSING_INDEX: u32 = 0xFFFFFFFF;

/// The push constants to be used with the ray tracing pipeline.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PushConstants {
    pub view_inverse: glm::Mat4,
    pub time: f32,
}

/// The camera lens uniform data to be used with the ray tracing pipeline.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CameraLens {
    pub projection_inverse: glm::Mat4,
}

/// The various shader resources needed for ray tracing.
struct RayTracingShaders {
    modules: [ash::vk::ShaderModule; 4],
}
impl RayTracingShaders {
    pub const RAY_GEN: u32 = 0;
    pub const MISS: u32 = 1;
    pub const SHADOW_MISS: u32 = 2;
    pub const CLOSEST_HIT: u32 = 3;

    /// Create a new instance of `RayTracingShaders`.
    pub fn new(device: &ash::Device) -> Self {
        let ray_gen = create_shader_module(
            device,
            inline_spirv::include_spirv!(
                "src/shaders/raytracing/raygen.rgen",
                rgen,
                glsl,
                vulkan1_2
            ),
        );
        let miss = create_shader_module(
            device,
            inline_spirv::include_spirv!(
                "src/shaders/raytracing/miss.rmiss",
                rmiss,
                glsl,
                vulkan1_2
            ),
        );
        let shadow_miss = create_shader_module(
            device,
            inline_spirv::include_spirv!(
                "src/shaders/raytracing/shadow.rmiss",
                rmiss,
                glsl,
                vulkan1_2
            ),
        );
        let closest_hit = create_shader_module(
            device,
            inline_spirv::include_spirv!(
                "src/shaders/raytracing/closest_hit.rchit",
                rchit,
                glsl,
                vulkan1_2
            ),
        );

        Self {
            modules: [ray_gen, miss, shadow_miss, closest_hit],
        }
    }

    /// Destroy the shader modules.
    pub fn destroy(&self, device: &ash::Device) {
        for module in self.modules {
            unsafe { device.destroy_shader_module(module, None) };
        }
    }

    // Getters.
    pub fn ray_gen(&self) -> ash::vk::ShaderModule {
        self.modules[Self::RAY_GEN as usize]
    }
    pub fn miss(&self) -> ash::vk::ShaderModule {
        self.modules[Self::MISS as usize]
    }
    pub fn shadow_miss(&self) -> ash::vk::ShaderModule {
        self.modules[Self::SHADOW_MISS as usize]
    }
    pub fn closest_hit(&self) -> ash::vk::ShaderModule {
        self.modules[Self::CLOSEST_HIT as usize]
    }
}

/// Minimal helper type for consistent texture pixel representations.
#[repr(C)]
struct TexturePixel {
    p: [u8; 4],
}
impl TexturePixel {
    pub fn new(p: [u8; 4]) -> Self {
        Self { p }
    }
}

/// Helper to allocate a texture on the GPU for each texture in a model.
fn allocate_gltf_textures(
    device: &ash::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    gltf_images: &[gltf::image::Data],
) -> Vec<utils::textures::AllocatedImage> {
    /// Determine the number of bytes in each glTF texture pixel given the input format.
    fn pixel_format_length(format: gltf::image::Format) -> usize {
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
    fn bytes_to_pixel(bytes: &[u8], source_format: gltf::image::Format) -> TexturePixel {
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
                pixel[0] =
                    (255. * f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])) as u8;
                pixel[1] =
                    (255. * f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])) as u8;
                pixel[2] =
                    (255. * f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]])) as u8;
            }
            Format::R32G32B32A32FLOAT => {
                pixel[0] =
                    (255. * f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])) as u8;
                pixel[1] =
                    (255. * f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])) as u8;
                pixel[2] =
                    (255. * f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]])) as u8;
                pixel[3] =
                    (255. * f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]])) as u8;
            }
        }
        TexturePixel::new(pixel)
    }

    // Determine the total number of bytes that are required to upload all the textures.
    let total_pixel_count: usize = gltf_images
        .iter()
        .map(|image| image.pixels.len() / pixel_format_length(image.format))
        .sum();

    #[cfg(debug_assertions)]
    println!(
        "INFO: Loading {} pixels across {} glTF textures",
        total_pixel_count,
        gltf_images.len(),
    );

    // Create a staging buffer to hold all the texture data.
    let mut staging_buffer = utils::buffers::StagingBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        total_pixel_count as u64 * std::mem::size_of::<TexturePixel>() as u64,
    )
    .expect("Failed to create staging buffer for glTF textures");

    // Convert and upload each image to the GPU.
    let mut staging_offset = 0;
    let (textures, fences, cleanup): (Vec<_>, Vec<_>, Vec<_>) =
        itertools::multiunzip(gltf_images.iter().map(|image| {
            // Ensure the image is converted into the approprite pixel format for the image.
            // NOTE: Here, we assume that the image is in the R8G8B8A8 format.
            let pixel_iter = image.pixels.chunks_exact(pixel_format_length(image.format));
            let pixels = pixel_iter
                .map(|chunk| bytes_to_pixel(chunk, image.format))
                .collect::<Vec<_>>();

            // Create a new (empty) image on the GPU with the appropriate size and usage flags.
            let texture = utils::textures::AllocatedImage::new(
                device,
                allocator,
                ash::vk::ImageCreateInfo::default()
                    .image_type(ash::vk::ImageType::TYPE_2D)
                    .format(ash::vk::Format::R8G8B8A8_UNORM)
                    .extent(ash::vk::Extent3D {
                        width: image.width,
                        height: image.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(ash::vk::SampleCountFlags::TYPE_1)
                    .tiling(ash::vk::ImageTiling::OPTIMAL)
                    .usage(
                        ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(ash::vk::SharingMode::EXCLUSIVE),
                None,
                "glTF Texture",
            );

            // Copy the image data to the GPU.
            let utils::CleanableFence { fence, cleanup } = utils::textures::copy_buffer_to_image(
                device,
                command_pool,
                queue,
                texture.image,
                ash::vk::Extent2D {
                    width: image.width,
                    height: image.height,
                },
                ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                utils::data_slice_byte_slice(&pixels),
                &mut staging_buffer,
                Some(staging_offset),
            )
            .expect("Failed to copy glTF texture to image");

            // Update the staging offset for the next texture.
            staging_offset += image.pixels.len();

            (texture, fence, cleanup)
        }));

    // Wait for all the fences to signal that the textures are ready, then delete the fences.
    // TODO: Return the fences and cleanup to the caller to allow for greater parallelism.
    unsafe { device.wait_for_fences(&fences, true, FIVE_SECONDS_IN_NANOSECONDS) }
        .expect("Failed to wait for all glTF texture upload fences");
    unsafe {
        for fence in fences {
            device.destroy_fence(fence, None);
        }
    }
    for cleanup in cleanup.into_iter().filter_map(std::convert::identity) {
        cleanup(device, allocator);
    }
    staging_buffer.destroy(device, allocator);

    textures
}

/// Create a sampler from a glTF sampler's description.
fn create_sampler_from_glft(
    device: &ash::Device,
    sampler_info: &gltf::texture::Sampler,
) -> ash::vk::Sampler {
    fn address_mode(mode: gltf::texture::WrappingMode) -> ash::vk::SamplerAddressMode {
        match mode {
            gltf::texture::WrappingMode::ClampToEdge => ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => {
                ash::vk::SamplerAddressMode::MIRRORED_REPEAT
            }
            gltf::texture::WrappingMode::Repeat => ash::vk::SamplerAddressMode::REPEAT,
        }
    }

    // Create a sampler from the glTF sampler description.
    let sampler_info = ash::vk::SamplerCreateInfo::default()
        .mag_filter(
            sampler_info
                .mag_filter()
                .map_or(ash::vk::Filter::LINEAR, |f| match f {
                    gltf::texture::MagFilter::Nearest => ash::vk::Filter::NEAREST,
                    gltf::texture::MagFilter::Linear => ash::vk::Filter::LINEAR,
                }),
        )
        .min_filter(
            sampler_info
                .min_filter()
                .map_or(ash::vk::Filter::LINEAR, |f| match f {
                    gltf::texture::MinFilter::Nearest => ash::vk::Filter::NEAREST,
                    gltf::texture::MinFilter::Linear => ash::vk::Filter::LINEAR,
                    gltf::texture::MinFilter::NearestMipmapNearest => ash::vk::Filter::NEAREST,
                    gltf::texture::MinFilter::LinearMipmapNearest => ash::vk::Filter::LINEAR,
                    gltf::texture::MinFilter::NearestMipmapLinear => ash::vk::Filter::NEAREST,
                    gltf::texture::MinFilter::LinearMipmapLinear => ash::vk::Filter::LINEAR,
                }),
        )
        .address_mode_u(address_mode(sampler_info.wrap_s()))
        .address_mode_v(address_mode(sampler_info.wrap_t()));

    unsafe { device.create_sampler(&sampler_info, None) }
        .expect("Failed to create sampler for glTF texture")
}

/// The data for a single geometry node.
/// TODO: Store each material in a different structure and reference which material index to use.
#[repr(C)]
struct GeometryNode {
    /// The base color to apply to the material.
    pub base_color: glm::Vec4,

    /// This is the starting index of the geometry for this node.
    pub index_offset: u32,

    /// The index of the texture to use for the base color.
    pub texture_color_index: u32,

    /// The index of the sampler to apply to the textures.
    pub color_sampler_index: u32,

    /// The base metallic factor to apply to the material.
    pub metallic_factor: f32,

    /// The base roughness factor to apply to the material.
    pub roughness_factor: f32,

    /// The index of the texture to use for normal mapping.
    pub texture_normal_index: u32,

    /// The index of the metallic-roughness texture of the material.
    pub texture_metallic_roughness_index: u32,

    /// Padding to ensure that the structure is a multiple of 16 bytes.
    _padding: [u32; 1],
}

/// Simple helper for managing the data for a mesh.
struct MeshGpuData {
    // Mesh data.
    pub vertex_buffer: utils::buffers::BufferAllocation,
    pub vertex_address: ash::vk::DeviceAddress,
    pub index_buffer: utils::buffers::BufferAllocation,
    pub index_address: ash::vk::DeviceAddress,
    pub max_vertex_index: u32,

    // Geometry nodes.
    pub geometry_node_buffer: utils::buffers::BufferAllocation,
    pub geometry_node_address: ash::vk::DeviceAddress,
    pub geometry_nodes: Vec<GeometryNode>,
    pub geometry_node_lengths: Vec<u64>,

    // Textures.
    pub textures: Vec<utils::textures::AllocatedImage>,
    pub samplers: Vec<ash::vk::Sampler>,
}
impl MeshGpuData {
    /// Destroy the mesh data.
    pub fn destroy(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        // Destroy the geometry buffers.
        self.geometry_node_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);

        // Destroy the textures.
        for texture in self.textures {
            texture.destroy(device, allocator);
        }
        for sampler in self.samplers {
            unsafe { device.destroy_sampler(sampler, None) };
        }
    }
}

/// The vertex data of meshes.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Vertex {
    position: glm::Vec3,
    normal: glm::Vec3,
    uv: glm::Vec2,
    tangent: glm::Vec4,
}

/// Traverse the glTF scene graph recursively to build the geometry nodes.
/// Use one geometry for each node in the mesh. This allows each node to have their material and
/// textures respected when rendering.
fn traverse_scene_nodes(
    device: &ash::Device,
    gltf_buffers: &[gltf::buffer::Data],
    node: &gltf::Node,
    parent_transform: &glm::Mat4,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    geometry_nodes: &mut Vec<GeometryNode>,
    geometry_node_lengths: &mut Vec<u64>,
    vertex_offset: &mut u32,
    samplers: &mut Vec<ash::vk::Sampler>,
) {
    // Determine the transformation matrix for this node.
    let t = match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => glm::Mat4::from(matrix),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let translation = glm::Vec3::new(translation[0], translation[1], translation[2]);
            let rotation = glm::quat(rotation[0], rotation[1], rotation[2], rotation[3]);
            let scale = glm::Vec3::new(scale[0], scale[1], scale[2]);

            glm::translation(&translation) * glm::quat_to_mat4(&rotation) * glm::scaling(&scale)
        }
    } * parent_transform;
    let t_inv_transpose = glm::Mat4::try_inverse(t)
        .map(|mut m| {
            m.transpose_mut();
            m
        })
        .unwrap_or_else(glm::Mat4::identity);

    // Traverse the children of this node, using the current transform as their parent.
    node.children().for_each(|n| {
        traverse_scene_nodes(
            device,
            gltf_buffers,
            &n,
            &t,
            vertices,
            indices,
            geometry_nodes,
            geometry_node_lengths,
            vertex_offset,
            samplers,
        );
    });

    // Add all triangle primitives to the scene, if any.
    let Some(mesh) = node.mesh() else {
        return;
    };
    mesh.primitives().for_each(|primitive| {
        // Determine if this primitive will be skipped.
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            println!("WARN: Skipping non-triangle primitive");
            return;
        }

        // Determine the base color of the material.
        let material = primitive.material();
        let base_color = material.pbr_metallic_roughness().base_color_factor().into();

        // Helper to get the sampler index or create a new one.
        let mut get_sampler_or_new = |sampler: &gltf::texture::Sampler| {
            sampler.index().unwrap_or_else(|| {
                let index = samplers.len();
                samplers.push(create_sampler_from_glft(device, sampler));
                index
            }) as u32
        };

        // Determine the texture indices for this primitive.
        let color_texture = material
            .pbr_metallic_roughness()
            .base_color_texture()
            .map(|t| {
                (
                    t.texture().source().index() as u32,
                    t.tex_coord(),
                    get_sampler_or_new(&t.texture().sampler()),
                )
            });
        let color_sampler_index = color_texture
            .as_ref()
            .map_or(MISSING_INDEX, |(_, _, sampler)| *sampler);

        // The base metallic and roughness factors for the material.
        let metallic_factor = material.pbr_metallic_roughness().metallic_factor();
        let roughness_factor = material.pbr_metallic_roughness().roughness_factor();

        // TODO: Use independent samplers for these textures.
        let texture_normal_index = material
            .normal_texture()
            .map_or(MISSING_INDEX, |t| t.texture().source().index() as u32);
        let texture_metallic_roughness_index = material
            .pbr_metallic_roughness()
            .metallic_roughness_texture()
            .map_or(MISSING_INDEX, |t| t.texture().source().index() as u32);

        // Push what we know about this geometry node.
        geometry_nodes.push(GeometryNode {
            base_color,
            index_offset: indices.len() as u32,
            texture_color_index: color_texture
                .as_ref()
                .map_or(MISSING_INDEX, |(texture_index, _, _)| *texture_index),
            color_sampler_index,
            metallic_factor,
            roughness_factor,
            texture_normal_index,
            texture_metallic_roughness_index,
            _padding: [0; 1],
        });

        // Read the data for this primitive.
        let reader = primitive.reader(|buffer| Some(&gltf_buffers[buffer.index()]));
        let positions = reader.read_positions().expect("Failed to read positions");
        let normals = reader.read_normals().expect("Failed to read normals");
        let mut tangents = reader.read_tangents();
        let mut uvs = color_texture
            .and_then(|(_, uv_set, _)| reader.read_tex_coords(uv_set))
            .map(|uv| uv.into_f32());
        let read_indices = reader.read_indices().expect("Failed to read indices");

        // Determine the number of new vertices and append them to the list.
        let mut position_count = 0;
        for (pos, norm) in positions.zip(normals) {
            position_count += 1;

            // Apply the node transformation to each vertex.
            let position = (t * glm::Vec4::new(pos[0], pos[1], pos[2], 1.)).xyz();
            let normal = (t_inv_transpose * glm::Vec4::new(norm[0], norm[1], norm[2], 0.)).xyz();
            let tangent = if let Some(tans) = tangents.as_mut().and_then(|t| t.next()) {
                t_inv_transpose * glm::Vec4::new(tans[0], tans[1], tans[2], tans[3])
            } else {
                glm::Vec4::zeros()
            };

            let uv = if let Some(uv) = uvs.as_mut().and_then(|u| u.next()) {
                uv.into()
            } else {
                glm::Vec2::zeros()
            };
            vertices.push(Vertex {
                position,
                normal,
                tangent,
                uv,
            });
        }

        // Append the indices for this primitive to the list.
        let mut index_count = 0;
        read_indices.into_u32().for_each(|i| {
            index_count += 1;
            indices.push(i + *vertex_offset);
        });

        *vertex_offset += position_count;
        geometry_node_lengths.push(index_count);
    });
}

/// Helper to create minimal geometry for this simple example.
//  TODO: Accept a texture index offset in case their are textures already by the engine.
fn create_mesh_data(
    device: &ash::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
) -> MeshGpuData {
    // Load the test model from memory.
    static DUCK_BYTES: &[u8] = include_bytes!("../../assets/Duck.glb");
    static PLANES_BYTES: &[u8] = include_bytes!("../../assets/Planes.glb");
    let bistro_bytes = std::fs::read("assets/Bistro.glb").expect("Failed to read Bistro.glb");
    let (test_gltf, gltf_buffers, gltf_images) = {
        // gltf::import_slice(DUCK_BYTES).expect("Failed to load test GLB model")
        // gltf::import_slice(PLANES_BYTES).expect("Failed to load test GLB model")
        gltf::import_slice(bistro_bytes).expect("Failed to load test GLB model")
    };

    // Create a sampler for each sampler description in the model.
    let mut samplers = test_gltf
        .samplers()
        .map(|s| create_sampler_from_glft(device, &s))
        .collect::<Vec<_>>();

    // Load the textures for the model.
    let textures = if gltf_images.is_empty() {
        Vec::new()
    } else {
        allocate_gltf_textures(
            device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            &gltf_images,
        )
    };

    // Setup buffers to hold the model data.
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut geometry_nodes = Vec::new();
    let mut geometry_node_lengths = Vec::new();
    let mut vertex_offset = 0;

    // Get the default scene for this model.
    let scene = test_gltf.default_scene().unwrap_or_else(|| {
        test_gltf
            .scenes()
            .next()
            .expect("Failed to find a default scene")
    });

    // Flip the Y-axis of glTF models to align with Vulkan coordinates.
    let root_transform = glm::scaling(&glm::Vec3::new(1., -1., 1.));

    // Traverse the scene graph to build the geometry nodes.
    scene.nodes().for_each(|node| {
        traverse_scene_nodes(
            device,
            &gltf_buffers,
            &node,
            &root_transform,
            &mut vertices,
            &mut indices,
            &mut geometry_nodes,
            &mut geometry_node_lengths,
            &mut vertex_offset,
            &mut samplers,
        );
    });

    #[cfg(debug_assertions)]
    assert!(
        !vertices.is_empty() && !indices.is_empty(),
        "Failed to load any vertices or indices from the glTF model"
    );

    // Create a staging buffer capable of holding each of the vertex, index, and transform data.
    let vertex_data = utils::data_slice_byte_slice(&vertices);
    let index_data = utils::data_slice_byte_slice(&indices);
    let gltf_instance_byte_length =
        2 * std::mem::size_of::<u64>() + std::mem::size_of::<GeometryNode>() * geometry_nodes.len();

    #[cfg(debug_assertions)]
    println!(
        "INFO: Loaded {} vertices and {} indices",
        vertices.len(),
        indices.len()
    );

    let max_buffer_size = (vertex_data.len() + index_data.len() + gltf_instance_byte_length) as u64;
    let mut staging_buffer = utils::buffers::StagingBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        max_buffer_size,
    )
    .expect("Failed to create staging buffer for bottom-level acceleration structure");

    // Each buffer requires a device address as well as usage during the acceleration structure build.
    let required_buffer_usage =
        ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

    // New device-local buffer for the vertex data.
    let (vertex_buffer, vertex_fence) = utils::buffers::new_data_buffer(
        device,
        pageable_device_local_memory.map(|d| (d, 1.)),
        allocator,
        command_pool,
        queue,
        vertex_data,
        required_buffer_usage,
        "Vertex Buffer",
        &mut staging_buffer,
        None,
    )
    .expect("Failed to create vertex buffer for bottom-level acceleration structure");
    let vertex_address = unsafe {
        device.get_buffer_device_address(
            &ash::vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
        )
    };

    // New device-local buffer for the index data.
    let (index_buffer, index_fence) = utils::buffers::new_data_buffer(
        device,
        pageable_device_local_memory.map(|d| (d, 1.)),
        allocator,
        command_pool,
        queue,
        index_data,
        required_buffer_usage,
        "Index Buffer",
        &mut staging_buffer,
        Some(vertex_data.len()),
    )
    .expect("Failed to create index buffer for bottom-level acceleration structure");
    let index_address = unsafe {
        device.get_buffer_device_address(
            &ash::vk::BufferDeviceAddressInfo::default().buffer(index_buffer.buffer),
        )
    };

    // New device-local buffer for the geometry nodes.
    let (geometry_node_buffer, geometry_node_fence) = {
        let gltf_instance_bytes = utils::data_byte_slice(&vertex_address)
            .iter()
            .copied()
            .chain(utils::data_byte_slice(&index_address).iter().copied())
            .chain(
                utils::data_slice_byte_slice(&geometry_nodes)
                    .iter()
                    .copied(),
            )
            .collect::<Vec<u8>>();

        utils::buffers::new_data_buffer(
            device,
            pageable_device_local_memory.map(|d| (d, 1.)),
            allocator,
            command_pool,
            queue,
            &gltf_instance_bytes,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Geometry Node Buffer",
            &mut staging_buffer,
            Some(vertex_data.len() + index_data.len()),
        )
        .expect("Failed to create geometry node buffer for bottom-level acceleration structure")
    };
    let geometry_node_address = unsafe {
        device.get_buffer_device_address(
            &ash::vk::BufferDeviceAddressInfo::default().buffer(geometry_node_buffer.buffer),
        )
    };

    // Wait for the fences to signal that the buffers are ready, then delete the fences.
    unsafe {
        device.wait_for_fences(
            &[
                vertex_fence.fence,
                index_fence.fence,
                geometry_node_fence.fence,
            ],
            true,
            FIVE_SECONDS_IN_NANOSECONDS,
        )
    }
    .expect("Failed to wait for buffer creation fences");

    // Clean up the staging buffer and fences.
    staging_buffer.destroy(device, allocator);
    vertex_fence.cleanup(device, allocator);
    index_fence.cleanup(device, allocator);
    geometry_node_fence.cleanup(device, allocator);

    MeshGpuData {
        vertex_buffer,
        vertex_address,
        index_buffer,
        index_address,
        max_vertex_index: (vertices.len() - 1) as u32,
        geometry_node_buffer,
        geometry_node_address,
        geometry_nodes,
        geometry_node_lengths,
        textures,
        samplers,
    }
}

/// Helper for instancing a mesh which is managed elsewhere, `MeshData`.
#[derive(Clone, Copy)]
struct AcceleratedInstance {
    /// A bottom-level acceleration structure for the mesh.
    pub acceleration_address: ash::vk::DeviceAddress,

    /// The global transformation to apply to this instance.
    pub transform: ash::vk::TransformMatrixKHR,
}

/// Create a new instance of a mesh.
fn create_mesh_instance(
    acceleration_address: ash::vk::DeviceAddress,
    transform: ash::vk::TransformMatrixKHR,
) -> AcceleratedInstance {
    AcceleratedInstance {
        acceleration_address,
        transform,
    }
}

/// Create a bottom-level acceleration structure.
/// # Safety
/// The `queue` must support compute operations.
fn create_bottom_level_acceleration_structure(
    device: &ash::Device,
    acceleration_device: &ash::khr::acceleration_structure::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    mesh: &MeshGpuData,
) -> utils::ray_tracing::AccelerationStructure {
    // Define the triangle geometry for the bottom-level acceleration structure.
    let (acceleration_geometries, geometry_counts) = mesh
        .geometry_nodes
        .iter()
        .zip(mesh.geometry_node_lengths.iter())
        .map(|(node, index_length)| {
            // Ensure that each geometry node is in a different geometry index in the acceleration structure.
            // This is required to allow a single BLAS to constain multiple materials and textures.
            let triangle_geometry_data = ash::vk::AccelerationStructureGeometryDataKHR {
                triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(ash::vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(ash::vk::DeviceOrHostAddressConstKHR {
                        device_address: mesh.vertex_address,
                    })
                    .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                    .max_vertex(mesh.max_vertex_index)
                    .index_type(ash::vk::IndexType::UINT32)
                    .index_data(ash::vk::DeviceOrHostAddressConstKHR {
                        device_address: mesh.index_address
                            + node.index_offset as u64 * std::mem::size_of::<u32>() as u64,
                    }),
            };
            let node_geometry = ash::vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
                .geometry(triangle_geometry_data)
                .flags(ash::vk::GeometryFlagsKHR::OPAQUE);

            (node_geometry, (index_length / 3) as u32)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    // Describe the total geometry of the acceleration structure to be built.
    let build_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&acceleration_geometries);

    #[cfg(debug_assertions)]
    println!(
        "INFO: Building acceleration structure with {} triangles across {} geometry nodes",
        geometry_counts.iter().map(|c| *c as u64).sum::<u64>(),
        geometry_counts.len(),
    );

    // Get the build size of this bottom-level acceleration structure.
    let mut acceleration_build_size = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        acceleration_device.get_acceleration_structure_build_sizes(
            ash::vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &geometry_counts,
            &mut acceleration_build_size,
        );
    }

    // Create a buffer that can be used as the scratch buffer for building the acceleration structure.
    // TODO: Allow calling this function with an optional scratch buffer and check if the existing
    // size is large enough for reuse.
    let scratch_buffer = utils::ray_tracing::ScratchBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        acceleration_build_size.build_scratch_size,
    );

    // Build the acceleration structure.
    let acceleration = utils::ray_tracing::AccelerationStructure::new_bottom_level(
        device,
        acceleration_device,
        pageable_device_local_memory,
        allocator,
        &acceleration_build_size,
        &scratch_buffer,
        utils::ray_tracing::InstancedGeometries::new(&acceleration_geometries, &geometry_counts)
            .unwrap(),
        command_pool,
        queue,
    );

    // Clean up the acceleration scratch buffer.
    scratch_buffer.destroy(device, allocator);

    acceleration
}

/// Create the top-level acceleration structure for this scene.
fn create_top_level_acceleration_structure(
    device: &ash::Device,
    acceleration_device: &ash::khr::acceleration_structure::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    mesh_instances: &[AcceleratedInstance],
) -> utils::ray_tracing::AccelerationStructure {
    // Determine the number of acceleration instances to create.
    let instance_count = mesh_instances.len() as u32;

    // Create a staging buffer to upload the data for each instance.
    let mut staging_buffer = utils::buffers::StagingBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        std::mem::size_of::<ash::vk::AccelerationStructureInstanceKHR>() as u64
            * instance_count as u64,
    )
    .expect("Failed to create staging buffer for acceleration structure instances");

    // The instance flags are used to control how the geometry is traversed.
    // TODO: Determine if we want to cull using this step.
    #[allow(clippy::cast_possible_truncation)]
    let instance_flag_as_u8 =
        ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8;

    // Create an instance for each bottom-level acceleration structure.
    let instances = mesh_instances
        .iter()
        .map(|mesh_instance| {
            // Create an instance of this bottom-level acceleration structure.
            ash::vk::AccelerationStructureInstanceKHR {
                transform: mesh_instance.transform,
                instance_custom_index_and_mask: ash::vk::Packed24_8::new(0, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: ash::vk::Packed24_8::new(
                    0,
                    instance_flag_as_u8,
                ),
                acceleration_structure_reference: ash::vk::AccelerationStructureReferenceKHR {
                    device_handle: mesh_instance.acceleration_address,
                },
            }
        })
        .collect::<Vec<_>>();

    // Create a device-local buffer for this instance.
    // TODO: Create a single buffer and index into it for each instance.
    let (instance_buffer, instance_fence) = utils::buffers::new_data_buffer(
        device,
        pageable_device_local_memory.map(|d| (d, 0.5)),
        allocator,
        command_pool,
        queue,
        utils::data_slice_byte_slice(&instances),
        ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "Instance Buffer",
        &mut staging_buffer,
        None,
    )
    .expect("Failed to create instance buffer for top-level acceleration structure");
    let instance_address = unsafe {
        device.get_buffer_device_address(
            &ash::vk::BufferDeviceAddressInfo::default().buffer(instance_buffer.buffer),
        )
    };

    // Wait for the fence to signal that the buffer is ready, then delete the fence.
    unsafe { device.wait_for_fences(&[instance_fence.fence], true, FIVE_SECONDS_IN_NANOSECONDS) }
        .expect("Failed to wait for all instance buffer creation fences");
    instance_fence.cleanup(device, allocator);

    // TODO: Allow reuse of staging buffer between functions.
    staging_buffer.destroy(device, allocator);

    // Describe the geometry and instancing for each bottom-level acceleration structure.
    let acceleration_geometry_instances = {
        // Point to our newly created instance buffer for this top-level geometry.
        let instances = ash::vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(ash::vk::DeviceOrHostAddressConstKHR {
                device_address: instance_address,
            });

        ash::vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR { instances })
            .flags(ash::vk::GeometryFlagsKHR::OPAQUE)
    };

    // Describe the total geometry of the acceleration structure to be built.
    let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(std::slice::from_ref(&acceleration_geometry_instances));

    // Get the build size of this top-level acceleration structure.
    let acceleration_build_sizes = unsafe {
        let mut build_sizes = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
        acceleration_device.get_acceleration_structure_build_sizes(
            ash::vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[instance_count],
            &mut build_sizes,
        );
        build_sizes
    };

    // Create a scratch buffer to be used during the acceleration structure build.
    // TODO: Allow calling this function with an optional scratch buffer and check if it can be reused.
    let scratch_buffer = utils::ray_tracing::ScratchBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        acceleration_build_sizes.build_scratch_size,
    );

    let acceleration = utils::ray_tracing::AccelerationStructure::new_top_level(
        device,
        acceleration_device,
        pageable_device_local_memory,
        allocator,
        &acceleration_build_sizes,
        &scratch_buffer,
        &acceleration_geometry_instances,
        instance_count,
        command_pool,
        queue,
    );

    // Clean up the acceleration scratch buffer.
    scratch_buffer.destroy(device, allocator);

    // Delete each instancing buffer.
    instance_buffer.destroy(device, allocator);

    acceleration
}

/// The descriptor set layouts for the ray tracing pipeline.
#[repr(C)]
struct DescriptorSetLayouts {
    pub acceleration: ash::vk::DescriptorSetLayout,
    pub output_image: ash::vk::DescriptorSetLayout,
    pub camera_lens: ash::vk::DescriptorSetLayout,
    pub model_data: ash::vk::DescriptorSetLayout,
}
impl DescriptorSetLayouts {
    /// Create the descriptor set layout for the ray tracing pipeline.
    fn new(device: &ash::Device) -> DescriptorSetLayouts {
        // Create the layout for the acceleration structure set.
        let acceleration_binding = [ash::vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(
                ash::vk::ShaderStageFlags::RAYGEN_KHR | ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )];
        let acceleration = unsafe {
            device.create_descriptor_set_layout(
                &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&acceleration_binding),
                None,
            )
        }
        .expect("Failed to create acceleration descriptor set layout for the ray tracing demo");

        // Create the layout for the output image set.
        let storage_image_binding = [ash::vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(ash::vk::ShaderStageFlags::RAYGEN_KHR)];
        let output_image = unsafe {
            device.create_descriptor_set_layout(
                &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&storage_image_binding),
                None,
            )
        }
        .expect("Failed to create image descriptor set layout for the ray tracing demo");

        // Create the layout for the camera lens uniform data.
        let camera_lens_binding = [ash::vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ash::vk::ShaderStageFlags::RAYGEN_KHR)];
        let camera_lens = unsafe {
            device.create_descriptor_set_layout(
                &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&camera_lens_binding),
                None,
            )
        }
        .expect("Failed to create camera lens descriptor set layout for the ray tracing demo");

        // Create the layout for the mesh data.
        let instances_binding = ash::vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        let textures_binding = ash::vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(ash::vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(MAX_TEXTURES)
            .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        let samplers_binding = ash::vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(ash::vk::DescriptorType::SAMPLER)
            .descriptor_count(MAX_TEXTURES)
            .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        let model_data = unsafe {
            device.create_descriptor_set_layout(
                &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    instances_binding,
                    textures_binding,
                    samplers_binding,
                ]),
                None,
            )
        }
        .expect("Failed to create model data descriptor set layout for the ray tracing demo");

        DescriptorSetLayouts {
            acceleration,
            output_image,
            camera_lens,
            model_data,
        }
    }

    /// Get the contents of this struct as a slice of `ash::vk::DescriptorSetLayout`.
    pub fn layouts(&self) -> &[ash::vk::DescriptorSetLayout] {
        unsafe {
            std::slice::from_raw_parts(
                &self.acceleration,
                std::mem::size_of::<DescriptorSetLayouts>()
                    / std::mem::size_of::<ash::vk::DescriptorSetLayout>(),
            )
        }
    }

    /// Destroy the descriptor set layouts.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            for layout in self.layouts() {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
    }
}

/// Create a new pipeline layout for the ray tracing demo.
fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layouts: &DescriptorSetLayouts,
) -> ash::vk::PipelineLayout {
    let push_constant_range = ash::vk::PushConstantRange::default()
        .stage_flags(ash::vk::ShaderStageFlags::RAYGEN_KHR)
        .size(std::mem::size_of::<PushConstants>() as u32);

    unsafe {
        device.create_pipeline_layout(
            &ash::vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(&[push_constant_range])
                .set_layouts(descriptor_set_layouts.layouts()),
            None,
        )
    }
    .expect("Failed to create ray tracing pipeline layout")
}

/// The shader binding tables for the ray tracing pipeline.
struct ShaderBindingTables {
    pub raygen: utils::buffers::BufferAllocation,
    pub raygen_address: ash::vk::DeviceAddress,
    pub miss: utils::buffers::BufferAllocation,
    pub miss_address: ash::vk::DeviceAddress,
    pub hit: utils::buffers::BufferAllocation,
    pub hit_address: ash::vk::DeviceAddress,
}
impl ShaderBindingTables {
    /// Create the shader binding tables for this ray tracing pipeline.
    fn new(
        device: &ash::Device,
        ray_device: &ash::khr::ray_tracing_pipeline::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        ray_pipeline: ash::vk::Pipeline,
        properties: &ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
    ) -> Self {
        let shader_group_count = 4; // TODO: Let the caller specify the layout and count of shader groups.
        let group_handle_size = properties.shader_group_handle_size as usize;
        let shader_group_handle_alignment = properties.shader_group_handle_alignment as usize;

        // Round up the size of the group handle to the nearest multiple of the alignment.
        let group_handle_size_aligned =
            utils::aligned_size(group_handle_size, shader_group_handle_alignment);
        let total_bytes = shader_group_count as usize * group_handle_size_aligned;

        // Create a `Vec<u8>` to hold the shader group handles in host memory.
        let shader_group_data = unsafe {
            ray_device.get_ray_tracing_shader_group_handles(
                ray_pipeline,
                0,
                shader_group_count,
                total_bytes,
            )
        }
        .expect("Failed to get ray tracing shader group handles");

        // Create a staging buffer for copying our shader group handles into device-local memory.
        let mut staging_buffer = utils::buffers::StagingBuffer::new(
            device,
            pageable_device_local_memory,
            allocator,
            total_bytes as u64,
        )
        .expect("Failed to create staging buffer for shader binding tables");

        // Create a buffer for each type of shader group.
        let buffer_usage = ash::vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let (raygen_binding_table, raygen_fence) = utils::buffers::new_data_buffer(
            device,
            pageable_device_local_memory.map(|d| (d, 1.)),
            allocator,
            command_pool,
            queue,
            &shader_group_data[..group_handle_size],
            buffer_usage,
            "Raygen Binding Table",
            &mut staging_buffer,
            None,
        )
        .expect("Failed to create raygen binding table");

        let mut offset = group_handle_size_aligned;
        let (miss_binding_table, miss_fence) = utils::buffers::new_data_buffer(
            device,
            pageable_device_local_memory.map(|d| (d, 1.)),
            allocator,
            command_pool,
            queue,
            &shader_group_data[offset..(offset + 2 * group_handle_size_aligned)],
            buffer_usage,
            "Miss Binding Table",
            &mut staging_buffer,
            Some(offset),
        )
        .expect("Failed to create miss binding table");

        offset = 3 * group_handle_size_aligned;
        let (hit_binding_table, hit_fence) = utils::buffers::new_data_buffer(
            device,
            pageable_device_local_memory.map(|d| (d, 1.)),
            allocator,
            command_pool,
            queue,
            &shader_group_data[offset..(offset + group_handle_size)],
            buffer_usage,
            "Hit Binding Table",
            &mut staging_buffer,
            Some(offset),
        )
        .expect("Failed to create hit binding table");

        // Get the device addresses for each of the buffers.
        let (raygen_address, miss_address, hit_address) = unsafe {
            let raygen = device.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(raygen_binding_table.buffer),
            );
            let miss = device.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(miss_binding_table.buffer),
            );
            let hit = device.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(hit_binding_table.buffer),
            );

            (raygen, miss, hit)
        };

        // Wait for the fences to signal that the buffers are ready, then delete the fences.
        unsafe {
            device
                .wait_for_fences(
                    &[raygen_fence.fence, miss_fence.fence, hit_fence.fence],
                    true,
                    FIVE_SECONDS_IN_NANOSECONDS,
                )
                .expect("Failed to wait for shader binding table creation fences");
            raygen_fence.cleanup(device, allocator);
            miss_fence.cleanup(device, allocator);
            hit_fence.cleanup(device, allocator);
        }

        // Clean up the staging buffer.
        staging_buffer.destroy(device, allocator);

        Self {
            raygen: raygen_binding_table,
            raygen_address,
            miss: miss_binding_table,
            miss_address,
            hit: hit_binding_table,
            hit_address,
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        self.raygen.destroy(device, allocator);
        self.miss.destroy(device, allocator);
        self.hit.destroy(device, allocator);
    }
}

/// A pipeline capable of performing hardware-accelerated ray tracing.
struct Pipeline {
    pub descriptor_set_layouts: DescriptorSetLayouts,
    pub layout: ash::vk::PipelineLayout,
    pub shaders: RayTracingShaders,
    pub pipeline: ash::vk::Pipeline,
}
impl Pipeline {
    pub fn new(device: &ash::Device, ray_device: &ash::khr::ray_tracing_pipeline::Device) -> Self {
        // Load the ray tracing shaders.
        let shaders = RayTracingShaders::new(device);

        // Create the descriptor set layouts describing the shader resource usage.
        let descriptor_set_layouts = DescriptorSetLayouts::new(device);
        let layout = create_pipeline_layout(device, &descriptor_set_layouts);

        let ray_gen_shader_info = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::RAYGEN_KHR)
            .module(shaders.ray_gen())
            .name(ENTRY_POINT_MAIN);
        let ray_gen_shader_group = ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(RayTracingShaders::RAY_GEN) // Index of the ray-gen shader.
            .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .intersection_shader(ash::vk::SHADER_UNUSED_KHR);

        let miss_shader_info = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::MISS_KHR)
            .module(shaders.miss())
            .name(ENTRY_POINT_MAIN);
        let miss_shader_group = ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(RayTracingShaders::MISS) // Index of the miss shader.
            .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .intersection_shader(ash::vk::SHADER_UNUSED_KHR);

        let shadow_miss_shader_info = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::MISS_KHR)
            .module(shaders.shadow_miss())
            .name(ENTRY_POINT_MAIN);
        let shadow_miss_shader_group = ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(RayTracingShaders::SHADOW_MISS) // Index of the shadow miss shader.
            .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .intersection_shader(ash::vk::SHADER_UNUSED_KHR);

        let closest_hit_shader_info = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(shaders.closest_hit())
            .name(ENTRY_POINT_MAIN);
        let closest_hit_shader_group = ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(ash::vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(RayTracingShaders::CLOSEST_HIT) // Index of the closest-hit shader.
            .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
            .intersection_shader(ash::vk::SHADER_UNUSED_KHR);

        // TODO: Use a function to ensure the correct order.
        let shader_stages = [
            ray_gen_shader_info,
            miss_shader_info,
            shadow_miss_shader_info,
            closest_hit_shader_info,
        ];
        let shader_groups = [
            ray_gen_shader_group,
            miss_shader_group,
            shadow_miss_shader_group,
            closest_hit_shader_group,
        ];

        // Describe the ray tracing pipeline to be created.
        let ray_tracing_pipeline_info = ash::vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&shader_stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(MAX_RAY_RECURSION_DEPTH)
            .layout(layout);

        // Create the ray tracing pipeline.
        let pipeline = unsafe {
            ray_device.create_ray_tracing_pipelines(
                ash::vk::DeferredOperationKHR::null(),
                ash::vk::PipelineCache::null(),
                &[ray_tracing_pipeline_info],
                None,
            )
        }
        .expect("Failed to create ray tracing pipeline")[0];

        Self {
            descriptor_set_layouts,
            layout,
            shaders,
            pipeline,
        }
    }

    /// Destroy the pipeline and its associated resources.
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
        self.descriptor_set_layouts.destroy(device);
        self.shaders.destroy(device);
    }
}

/// Create the uniform buffer for the camera lens data.
fn create_lens_buffer(
    device: &ash::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    extent: ash::vk::Extent2D,
) -> utils::buffers::BufferAllocation {
    let projection_inverse = create_projection_matrix(extent)
        .try_inverse()
        .unwrap_or_else(glm::Mat4::identity);

    let mut staging = utils::buffers::StagingBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        std::mem::size_of::<glm::Mat4>() as u64,
    )
    .expect("Failed to create staging buffer for camera lens data");

    let (camera_lens_buffer, camera_fence) = utils::buffers::new_data_buffer(
        device,
        pageable_device_local_memory.map(|d| (d, 0.8)), // The lens is used once per ray; it is high priority, but not maximally so.
        allocator,
        command_pool,
        queue,
        utils::data_byte_slice(&projection_inverse),
        ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
        "Camera Lens Buffer",
        &mut staging,
        None,
    )
    .expect("Failed to create camera lens buffer for the ray tracing demo");

    // Wait for the fence to signal that the buffer is ready, then delete the fence.
    unsafe { device.wait_for_fences(&[camera_fence.fence], true, FIVE_SECONDS_IN_NANOSECONDS) }
        .expect("Failed to wait for camera lens buffer creation fence");
    camera_fence.cleanup(device, allocator);

    // Clean up the staging buffer.
    staging.destroy(device, allocator);

    camera_lens_buffer
}

/// Create the storage buffer specifying the glTF model used for each instance.
fn create_instances_buffer(
    device: &ash::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    mesh_instances: &[ash::vk::DeviceAddress],
) -> utils::buffers::BufferAllocation {
    let mut staging = utils::buffers::StagingBuffer::new(
        device,
        pageable_device_local_memory,
        allocator,
        std::mem::size_of_val(mesh_instances) as u64,
    )
    .expect("Failed to create staging buffer for instance data");

    let (instances_buffer, instances_fence) = utils::buffers::new_data_buffer(
        device,
        pageable_device_local_memory.map(|d| (d, 0.8)), // The instances are used once per ray; they are high priority, but not maximally so.
        allocator,
        command_pool,
        queue,
        utils::data_slice_byte_slice(mesh_instances),
        ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "Instances Buffer",
        &mut staging,
        None,
    )
    .expect("Failed to create instances buffer for the ray tracing demo");

    // Wait for the fence to signal that the buffer is ready, then delete the fence.
    // TODO: Allow returning the fence for the caller to wait on.
    unsafe { device.wait_for_fences(&[instances_fence.fence], true, FIVE_SECONDS_IN_NANOSECONDS) }
        .expect("Failed to wait for instances buffer creation fence");
    instances_fence.cleanup(device, allocator);

    // Clean up the staging buffer.
    staging.destroy(device, allocator);

    instances_buffer
}

/// A helper type for managing the descriptor sets for the ray tracing pipeline.
struct Descriptors {
    pub descriptor_pool: ash::vk::DescriptorPool,
    pub acceleration_set: ash::vk::DescriptorSet,
    pub output_image_sets: Vec<ash::vk::DescriptorSet>,
    pub camera_lens_set: ash::vk::DescriptorSet,
    pub camera_lens_buffer: utils::buffers::BufferAllocation,
    pub model_data_set: ash::vk::DescriptorSet,
    pub instances_buffer: utils::buffers::BufferAllocation,
}
impl Descriptors {
    /// Destroy the descriptor pool and descriptor sets.
    pub fn destroy(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }

        // Destroy the camera lens uniform buffer.
        self.camera_lens_buffer.destroy(device, allocator);

        // Destroy the instances buffer.
        self.instances_buffer.destroy(device, allocator);
    }
}

/// Helper to create the projection matrix given a new window aspect ratio.
fn create_projection_matrix(extent: ash::vk::Extent2D) -> glm::Mat4 {
    glm::Mat4::new_perspective(
        extent.width as f32 / extent.height as f32,
        std::f32::consts::FRAC_PI_2 * (7.5 / 9.), // 75 degrees.
        0.1,
        512.,
    )
}

/// Create the descriptor sets to all the data needed to trace the scene and draw the output.
fn create_descriptor_sets<'a, I>(
    device: &ash::Device,
    output_images: &[ash::vk::ImageView],
    descriptor_set_layouts: &DescriptorSetLayouts,
    top_acceleration: &utils::ray_tracing::AccelerationStructure,
    camera_lens_buffer: utils::buffers::BufferAllocation,
    instances_buffer: utils::buffers::BufferAllocation,
    instance_count: usize,
    textures: I,
    samplers: &[ash::vk::Sampler],
) -> Descriptors
where
    I: IntoIterator<Item = &'a utils::textures::AllocatedImage>,
{
    let image_count = output_images.len() as u32;

    // Create the descriptor pool for the ray tracer.
    // TODO: Consider whether multiple top-level acceleration structures are needed.
    //       It could facilitate updating the scene in-flight, but I don't know for certain.
    // TODO: Calculate `max_sets` based on the `DescriptorSetLayouts`.
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(3 + image_count)
                .pool_sizes(&[
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        descriptor_count: 1,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: image_count,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 1,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: MAX_TEXTURES as u32,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::SAMPLER,
                        descriptor_count: MAX_TEXTURES as u32,
                    },
                ]),
            None,
        )
    }
    .expect("Failed to create ray tracer descriptor pool");

    // TODO: Reduce the number of `allocate_descriptor_sets` calls below by utilizing the array
    // based API.

    // Allocate the acceleration descriptor set.
    let acceleration_descriptor_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layouts.acceleration));
    let acceleration_set =
        unsafe { device.allocate_descriptor_sets(&acceleration_descriptor_info) }
            .expect("Failed to allocate acceleration descriptor set for the ray tracer")[0];

    // To avoid updating the output image descriptor set for each frame, we allocate one per frame.
    let image_descriptor_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layouts.output_image));
    let output_image_sets = output_images
        .iter()
        .map(|_| {
            // Create a descriptor set for the acceleration structure.
            unsafe { device.allocate_descriptor_sets(&image_descriptor_info) }
                .expect("Failed to allocate image descriptor sets for the ray tracer")[0]
        })
        .collect::<Vec<_>>();

    // Allocate the camera lens descriptor set.
    let camera_lens_descriptor_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layouts.camera_lens));
    let camera_lens_set = unsafe { device.allocate_descriptor_sets(&camera_lens_descriptor_info) }
        .expect("Failed to allocate camera lens descriptor set for the ray tracer")[0];

    // Allocate the model data descriptor set.
    let model_data_descriptor_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layouts.model_data));
    let model_data_set = unsafe { device.allocate_descriptor_sets(&model_data_descriptor_info) }
        .expect("Failed to allocate model data descriptor set for the ray tracer")[0];

    // Set the update data for the top-level acceleration structure.
    let acceleration_handles = [top_acceleration.handle()];
    let mut update_acceleration_info =
        ash::vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&acceleration_handles);
    let acceleration_write = ash::vk::WriteDescriptorSet::default()
        .dst_set(acceleration_set)
        .dst_binding(0)
        .descriptor_count(1)
        .descriptor_type(ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut update_acceleration_info);

    // Set the update data for the camera lens uniform.
    let camera_info = [ash::vk::DescriptorBufferInfo::default()
        .buffer(camera_lens_buffer.buffer)
        .range(std::mem::size_of::<CameraLens>() as u64)];
    let camera_lens_write = ash::vk::WriteDescriptorSet::default()
        .dst_set(camera_lens_set)
        .dst_binding(0)
        .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&camera_info);

    // Set the update data for the instances buffer and the textures.
    let instances_info = [ash::vk::DescriptorBufferInfo::default()
        .buffer(instances_buffer.buffer)
        .range(std::mem::size_of::<ash::vk::DeviceAddress>() as u64 * instance_count as u64)];
    let instances_write = ash::vk::WriteDescriptorSet::default()
        .dst_set(model_data_set)
        .dst_binding(0)
        .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&instances_info);
    let textures_info = textures
        .into_iter()
        .map(|texture| {
            ash::vk::DescriptorImageInfo::default()
                .image_view(texture.image_view)
                .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        })
        .collect::<Vec<_>>();
    let textures_write = ash::vk::WriteDescriptorSet::default()
        .dst_set(model_data_set)
        .dst_binding(1)
        .descriptor_type(ash::vk::DescriptorType::SAMPLED_IMAGE)
        .image_info(&textures_info);
    let samplers_info = samplers
        .iter()
        .map(|sampler| ash::vk::DescriptorImageInfo::default().sampler(*sampler))
        .collect::<Vec<_>>();
    let samplers_write = ash::vk::WriteDescriptorSet::default()
        .dst_set(model_data_set)
        .dst_binding(2)
        .descriptor_type(ash::vk::DescriptorType::SAMPLER)
        .image_info(&samplers_info);

    // Set the update data for each output image.
    let image_info = output_images
        .iter()
        .map(|image_view| {
            ash::vk::DescriptorImageInfo::default()
                .image_view(*image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL)
        })
        .collect::<Vec<_>>();
    let image_writes =
        image_info
            .iter()
            .zip(output_image_sets.iter())
            .map(|(image_info, image_set)| {
                // Update the bindings for each output image.
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(*image_set)
                    .dst_binding(0)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(image_info))
            });

    // Collect all of the descriptor set writes into a single vector.
    let descriptor_set_writes = std::iter::once(acceleration_write)
        .chain(std::iter::once(camera_lens_write))
        .chain(std::iter::once(instances_write))
        .chain(if textures_info.is_empty() {
            None
        } else {
            Some(textures_write)
        })
        .chain(if samplers_info.is_empty() {
            None
        } else {
            Some(samplers_write)
        })
        .chain(image_writes)
        .collect::<Vec<_>>();

    // Update the descriptor sets with the new bindings.
    unsafe {
        device.update_descriptor_sets(&descriptor_set_writes, &[]);
    }

    Descriptors {
        descriptor_pool,
        acceleration_set,
        output_image_sets,
        camera_lens_set,
        camera_lens_buffer,
        model_data_set,
        instances_buffer,
    }
}

/// An example demonstrating the use of ray tracing.
pub struct ExampleRayTracing {
    mesh: MeshGpuData,
    bottom_acceleration: utils::ray_tracing::AccelerationStructure,
    top_acceleration: utils::ray_tracing::AccelerationStructure,
    pipeline: Pipeline,
    shader_binding_tables: ShaderBindingTables,
    descriptors: Descriptors,
}
impl ExampleRayTracing {
    /// Create a new instance of `ExampleRayTracing`.
    pub fn new(
        device: &ash::Device,
        acceleration_device: &ash::khr::acceleration_structure::Device,
        ray_device: &ash::khr::ray_tracing_pipeline::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
        output_images: &[ash::vk::ImageView],
        extent: ash::vk::Extent2D,
        properties: &ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    ) -> Self {
        let mesh = create_mesh_data(
            device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
        );

        // Create the acceleration structures for the geometry of this scene.
        let bottom_acceleration = create_bottom_level_acceleration_structure(
            device,
            acceleration_device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            &mesh,
        );

        let mesh_instance = create_mesh_instance(
            bottom_acceleration.device_address(),
            ash::vk::TransformMatrixKHR {
                matrix: [1.2, 0., 0., 0., 0., 1.2, 0., 0., 0., 0., 1.2, 0.],
            },
        );
        let mesh_instance_1 = create_mesh_instance(
            bottom_acceleration.device_address(),
            ash::vk::TransformMatrixKHR {
                matrix: [0., 0.3, 0., 2., -0.3, 0., 0., 0., 0., 0., 0.3, 2.],
            },
        );
        let mesh_instance_2 = create_mesh_instance(
            bottom_acceleration.device_address(),
            ash::vk::TransformMatrixKHR {
                matrix: [0., 0., 0.4, -3., 0., 0.4, 0., 0., -0.4, 0., 0., -3.],
            },
        );
        // let (mesh_instances, mesh_instance_geometries) = (
        //     [mesh_instance, mesh_instance_1, mesh_instance_2],
        //     [mesh.geometry_node_address; 3]
        // );
        let (mesh_instances, mesh_instance_geometries) =
            ([mesh_instance], [mesh.geometry_node_address]);

        // Create a buffer pointing to the geometry nodes of each instance in the top-level acceleration structure.
        let instances_buffer = create_instances_buffer(
            device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            &mesh_instance_geometries,
        );

        // Create the top-level acceleration structure over the desired instances.
        let top_acceleration = create_top_level_acceleration_structure(
            device,
            acceleration_device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            &mesh_instances,
        );

        // Create the ray tracing pipeline.
        // NOTE: This is completely distinct from the graphics pipeline.
        //       In fact, it is more similar to a compute pipeline, but is distinct from that as well.
        let pipeline = Pipeline::new(device, ray_device);

        // Create the shader binding tables.
        let shader_binding_tables = ShaderBindingTables::new(
            device,
            ray_device,
            pageable_device_local_memory,
            allocator,
            pipeline.pipeline,
            properties,
            command_pool,
            queue,
        );

        // Create the uniform buffer storing the camera projection matrix.
        // Requires an update when the window size changes.
        let camera_lens = create_lens_buffer(
            device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            extent,
        );

        // Create the descriptor sets for the ray tracing pipeline.
        let descriptors = create_descriptor_sets(
            device,
            output_images,
            &pipeline.descriptor_set_layouts,
            &top_acceleration,
            camera_lens,
            instances_buffer,
            mesh_instances.len(),
            &mesh.textures,
            &mesh.samplers,
        );

        Self {
            mesh,
            bottom_acceleration,
            top_acceleration,
            pipeline,
            shader_binding_tables,
            descriptors,
        }
    }

    /// Destroy the resources used by this example.
    pub fn destroy(
        self,
        device: &ash::Device,
        acceleration_device: &ash::khr::acceleration_structure::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        self.descriptors.destroy(device, allocator);
        self.shader_binding_tables.destroy(device, allocator);
        self.pipeline.destroy(device);
        self.top_acceleration
            .destroy(device, acceleration_device, allocator);
        self.bottom_acceleration
            .destroy(device, acceleration_device, allocator);
        self.mesh.destroy(device, allocator);
    }

    /// Record the ray tracing command buffer for this frame.
    pub fn record_command_buffer(
        &self,
        device: &ash::Device,
        ray_device: &ash::khr::ray_tracing_pipeline::Device,
        command_buffer: ash::vk::CommandBuffer,
        pipeline_properties: &ash::vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        output_image: ash::vk::Image,
        image_index: u32,
        extent: ash::vk::Extent2D,
        push_constants: &PushConstants,
    ) {
        let image_range = ash::vk::ImageSubresourceRange::default()
            .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // TODO: Use `Descriptors` to generate this array?
        let descriptor_sets = [
            self.descriptors.acceleration_set,
            self.descriptors.output_image_sets[image_index as usize],
            self.descriptors.camera_lens_set,
            self.descriptors.model_data_set,
        ];

        // Describe to the command buffer how to access the shader binding tables.
        let handle_aligned_size = u64::from(utils::aligned_size(
            pipeline_properties.shader_group_handle_size,
            pipeline_properties.shader_group_handle_alignment,
        ));

        let raygen_sbt = ash::vk::StridedDeviceAddressRegionKHR::default()
            .device_address(self.shader_binding_tables.raygen_address)
            .stride(handle_aligned_size)
            .size(handle_aligned_size);
        let miss_sbt = ash::vk::StridedDeviceAddressRegionKHR::default()
            .device_address(self.shader_binding_tables.miss_address)
            .stride(handle_aligned_size)
            .size(2 * handle_aligned_size); // TODO: Allow getting the shader group count programmatically.
        let hit_sbt = ash::vk::StridedDeviceAddressRegionKHR::default()
            .device_address(self.shader_binding_tables.hit_address)
            .stride(handle_aligned_size)
            .size(handle_aligned_size);

        // Even though we are not using a callables shader, we need to give this struct to the
        // raytrace pipeline.
        let callable_sbt = ash::vk::StridedDeviceAddressRegionKHR::default();

        unsafe {
            // Use a pipeline barrier to transition the output image for general use.
            let image_barrier = ash::vk::ImageMemoryBarrier2::default()
                .src_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                .dst_stage_mask(ash::vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(ash::vk::AccessFlags2::SHADER_WRITE)
                .old_layout(ash::vk::ImageLayout::UNDEFINED)
                .new_layout(ash::vk::ImageLayout::GENERAL)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(output_image)
                .subresource_range(image_range);
            device.cmd_pipeline_barrier2(
                command_buffer,
                &ash::vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            );

            // Bind the ray tracing pipeline.
            device.cmd_bind_pipeline(
                command_buffer,
                ash::vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline.pipeline,
            );

            // Set the push constants.
            device.cmd_push_constants(
                command_buffer,
                self.pipeline.layout,
                ash::vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                utils::data_byte_slice(push_constants),
            );

            // Bind the descriptor sets for the ray tracing pipeline.
            device.cmd_bind_descriptor_sets(
                command_buffer,
                ash::vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline.layout,
                0,
                &descriptor_sets,
                &[],
            );

            ray_device.cmd_trace_rays(
                command_buffer,
                &raygen_sbt,
                &miss_sbt,
                &hit_sbt,
                &callable_sbt,
                extent.width,
                extent.height,
                1, // Depth
            );

            // Transition back to a presentation layout.
            let image_barrier = ash::vk::ImageMemoryBarrier2::default()
                .src_stage_mask(ash::vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .src_access_mask(ash::vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                .old_layout(ash::vk::ImageLayout::GENERAL)
                .new_layout(ash::vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(output_image)
                .subresource_range(image_range);
            device.cmd_pipeline_barrier2(
                command_buffer,
                &ash::vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            );
        }
    }

    /// Recreate the descriptor sets for the output images and camera perspective when they are recreated.
    /// Returns a fence that must be waited on before rendering the next frame.
    pub fn recreate_view_sets(
        &mut self,
        device: &ash::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
        output_images: &[ash::vk::ImageView],
        extent: ash::vk::Extent2D,
    ) -> utils::CleanableFence {
        // Describe the new image views to write to our descriptor sets.
        let image_info = output_images
            .iter()
            .map(|image_view| {
                ash::vk::DescriptorImageInfo::default()
                    .image_view(*image_view)
                    .image_layout(ash::vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        // Update the projection matrix in the camera lens buffer.
        let projection_inverse = create_projection_matrix(extent)
            .try_inverse()
            .unwrap_or_else(glm::Mat4::identity);
        let mut staging = utils::buffers::StagingBuffer::new(
            device,
            pageable_device_local_memory,
            allocator,
            std::mem::size_of::<glm::Mat4>() as u64,
        )
        .expect("Failed to create staging buffer for camera lens data");
        let utils::CleanableFence {
            fence: camera_update_fence,
            cleanup: camera_cmd_buffer_cleanup,
        } = utils::buffers::update_device_local(
            device,
            command_pool,
            queue,
            &self.descriptors.camera_lens_buffer,
            utils::data_byte_slice(&projection_inverse),
            &mut staging,
            None,
        )
        .expect("Failed to update camera lens buffer for the ray tracing demo");

        // Allow the caller to wait and free the resources used as needed.
        let update_fence = utils::CleanableFence::new(
            camera_update_fence,
            Some(Box::new(
                move |device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator| {
                    staging.destroy(device, allocator);
                    if let Some(cleanup) = camera_cmd_buffer_cleanup {
                        cleanup(device, allocator);
                    }
                },
            )),
        );

        // Create the descriptor write objects.
        let descriptor_set_writes = image_info
            .iter()
            .zip(self.descriptors.output_image_sets.iter())
            .map(|(image_info, image_set)| {
                // Update the bindings for each output image.
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(*image_set)
                    .dst_binding(0)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(image_info))
            })
            .collect::<Vec<_>>();

        // Update all the output image descriptor sets.
        unsafe {
            device.update_descriptor_sets(&descriptor_set_writes, &[]);
        }

        update_fence
    }
}
