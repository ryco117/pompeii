use crate::engine::utils::{
    self, create_shader_module, shaders::ENTRY_POINT_MAIN, FIVE_SECONDS_IN_NANOSECONDS,
};

use nalgebra_glm as glm;

/// The maximum number of traversal iterations to allow before a ray must terminate.
const MAX_RAY_RECURSION_DEPTH: u32 = 10;

/// The maximum number of textures that can be used in the ray tracing pipeline.
const MAX_TEXTURES: u32 = 1024;

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

/// Simple helper for managing the data for a mesh.
pub struct MeshGpuData {
    pub vertex_buffer: utils::buffers::BufferAllocation,
    pub vertex_address: ash::vk::DeviceAddress,
    pub index_buffer: utils::buffers::BufferAllocation,
    pub index_address: ash::vk::DeviceAddress,
    pub indices_length: u64,
    pub max_vertex_index: u32,
}
impl MeshGpuData {
    /// Destroy the mesh data.
    pub fn destroy(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        self.index_buffer.destroy(device, allocator);
        self.vertex_buffer.destroy(device, allocator);
    }
}

/// Helper for instancing a mesh which is managed elsewhere, `MeshData`.
#[derive(Clone, Copy)]
pub struct AcceleratedInstance {
    /// A bottom-level acceleration structure for the mesh.
    pub acceleration_address: ash::vk::DeviceAddress,

    /// The global transformation to apply to this instance.
    pub transform: ash::vk::TransformMatrixKHR,
}

/// The vertex data of meshes.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Vertex {
    position: glm::Vec3,
}

/// Helper to create minimal geometry for this simple example.
fn create_mesh_data(
    device: &ash::Device,
    pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
) -> MeshGpuData {
    // Load the test model.
    static DUCK_BYTES: &[u8] = include_bytes!("../../assets/Duck.glb");
    let test_gltf = gltf::import_slice(DUCK_BYTES).expect("Failed to load test GLB model");

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut index_offset = 0;
    test_gltf.0.nodes().for_each(|node| {
        let Some(mesh) = node.mesh() else {
            return;
        };

        // Determine the transformation matrix for this node and apply it.
        // NOTE: The glTF vertical axis needs to be flipped to match Vulkan's coordinate system.
        let t = match node.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                glm::Mat4::from(matrix) * glm::scaling(&glm::Vec3::new(1., -1., 1.))
            }
            gltf::scene::Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => {
                let translation = glm::Vec3::new(translation[0], translation[1], translation[2]);
                let rotation = glm::quat(rotation[0], rotation[1], rotation[2], rotation[3]);
                let scale = glm::Vec3::new(scale[0], -scale[1], scale[2]);
                glm::translation(&translation) * glm::quat_to_mat4(&rotation) * glm::scaling(&scale)
            }
        };

        mesh.primitives().for_each(|primitive| {
            let texture_index = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|t| t.texture().index());

            if primitive.mode() != gltf::mesh::Mode::Triangles {
                println!("Skipping non-triangle primitive");
                return;
            }

            // let material = primitive.material().index();
            // Read the data for this primitive.
            let reader = primitive.reader(|buffer| Some(&test_gltf.1[buffer.index()]));
            let positions = reader.read_positions().expect("Failed to read positions");
            // let normals = reader.read_normals().expect("Failed to read normals");
            let read_indices = reader.read_indices().expect("Failed to read indices");

            let mut position_count = 0;
            for [pos_x, pos_y, pos_z] in positions {
                position_count += 1;
                let position = (t * glm::Vec4::new(pos_x, pos_y, pos_z, 1.)).xyz();
                vertices.push(Vertex { position });
            }

            read_indices.into_u32().for_each(|i| {
                indices.push(i + index_offset);
            });
            index_offset += position_count;
        });
    });

    #[cfg(debug_assertions)]
    assert!(
        !vertices.is_empty() && !indices.is_empty(),
        "Failed to load any vertices or indices from the duck model"
    );

    // Create a staging buffer capable of holding each of the vertex, index, and transform data.
    let vertex_data = utils::data_slice_byte_slice(&vertices);
    let index_data = utils::data_slice_byte_slice(&indices);

    #[cfg(debug_assertions)]
    println!(
        "INFO: Loaded {} vertex bytes and {} index bytes",
        vertex_data.len(),
        index_data.len()
    );

    let max_buffer_size = (vertex_data.len() + index_data.len()) as u64;
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

    // Wait for the fences to signal that the buffers are ready, then delete the fences.
    unsafe {
        device.wait_for_fences(
            &[vertex_fence, index_fence],
            true,
            FIVE_SECONDS_IN_NANOSECONDS,
        )
    }
    .expect("Failed to wait for buffer creation fences");

    // Clean up the staging buffer and fences.
    staging_buffer.destroy(device, allocator);
    unsafe {
        device.destroy_fence(vertex_fence, None);
        device.destroy_fence(index_fence, None);
    }

    MeshGpuData {
        vertex_buffer,
        vertex_address,
        index_buffer,
        index_address,
        max_vertex_index: (vertices.len() - 1) as u32,
        indices_length: indices.len() as u64,
    }
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
    let mut triangle_geometry_data = ash::vk::AccelerationStructureGeometryDataKHR::default();
    triangle_geometry_data.triangles =
        ash::vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(ash::vk::Format::R32G32B32_SFLOAT)
            .vertex_data(ash::vk::DeviceOrHostAddressConstKHR {
                device_address: mesh.vertex_address,
            })
            .vertex_stride(std::mem::size_of::<Vertex>() as u64)
            .max_vertex(mesh.max_vertex_index)
            .index_type(ash::vk::IndexType::UINT32)
            .index_data(ash::vk::DeviceOrHostAddressConstKHR {
                device_address: mesh.index_address,
            });
    let acceleration_geometries = [ash::vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
        .geometry(triangle_geometry_data)
        .flags(ash::vk::GeometryFlagsKHR::OPAQUE)];

    let build_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&acceleration_geometries);

    // NOTE: There must be an entry here for each geometry in the `build_info.geometries()` array.
    let geometry_counts = [(mesh.indices_length / 3) as u32];

    #[cfg(debug_assertions)]
    println!(
        "INFO: Building acceleration structure with {} triangles",
        geometry_counts[0]
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
    let instance_count = mesh_instances.len();

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
    unsafe {
        device
            .wait_for_fences(&[instance_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
            .expect("Failed to wait for all instance buffer creation fences");
        device.destroy_fence(instance_fence, None);
    }

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

        [ash::vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR { instances })
            .flags(ash::vk::GeometryFlagsKHR::OPAQUE)]
    };

    // The number of instances to include in each geometry of `acceleration_geometry_instances`.
    let primitive_counts = [instance_count as u32];

    // Describe the total geometry of the acceleration structure to be built.
    let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&acceleration_geometry_instances);

    // Get the build size of this top-level acceleration structure.
    let acceleration_build_sizes = unsafe {
        let mut build_sizes = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
        acceleration_device.get_acceleration_structure_build_sizes(
            ash::vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &primitive_counts,
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
        &utils::ray_tracing::InstancedGeometries::new(
            &acceleration_geometry_instances,
            &primitive_counts,
        )
        .unwrap(),
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
    // pub textures: ash::vk::DescriptorSetLayout,
    // pub mesh: ash::vk::DescriptorSetLayout,
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
                ash::vk::ShaderStageFlags::RAYGEN_KHR /*TODO: | ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR*/,
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

        // // Create the layout for the scene textures and samplers.
        // let textures_binding = ash::vk::DescriptorSetLayoutBinding::default()
        //     .binding(0)
        //     .descriptor_type(ash::vk::DescriptorType::SAMPLED_IMAGE)
        //     .descriptor_count(MAX_TEXTURES)
        //     .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        // let samplers_binding = ash::vk::DescriptorSetLayoutBinding::default()
        //     .binding(1)
        //     .descriptor_type(ash::vk::DescriptorType::SAMPLER)
        //     .descriptor_count(MAX_TEXTURES)
        //     .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        // let textures = unsafe {
        //     device.create_descriptor_set_layout(
        //         &ash::vk::DescriptorSetLayoutCreateInfo::default()
        //             .bindings(&[textures_binding, samplers_binding]),
        //         None,
        //     )
        // }
        // .expect("Failed to create texture and sampler descriptor set layout for the ray tracing demo");

        // // Create the layout for the mesh data.
        // let mech_binding = ash::vk::DescriptorSetLayoutBinding::default()
        //     .binding(0)
        //     .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
        //     .descriptor_count(4) // TODO: I may do something different here to allow for a more
        //     // straight-forward approach to the mesh indexing.
        //     .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        // let mesh = unsafe {
        //     device.create_descriptor_set_layout(
        //         &ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&[mech_binding]),
        //         None,
        //     )
        // }
        // .expect("Failed to create mesh descriptor set layout for the ray tracing demo");

        DescriptorSetLayouts {
            acceleration,
            output_image,
            camera_lens,
            // textures,
            // mesh,
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
            device.destroy_descriptor_set_layout(self.acceleration, None);
            device.destroy_descriptor_set_layout(self.output_image, None);
            device.destroy_descriptor_set_layout(self.camera_lens, None);
            // device.destroy_descriptor_set_layout(self.textures, None);
            // device.destroy_descriptor_set_layout(self.mesh, None);
        }
    }
}

/// Create a new pipeline layout for the ray tracing demo.
fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layouts: &DescriptorSetLayouts,
) -> ash::vk::PipelineLayout {
    let push_constant_range = ash::vk::PushConstantRange::default()
        .stage_flags(
            ash::vk::ShaderStageFlags::RAYGEN_KHR /*TODO: | ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR*/,
        )
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
                    &[raygen_fence, miss_fence, hit_fence],
                    true,
                    FIVE_SECONDS_IN_NANOSECONDS,
                )
                .expect("Failed to wait for shader binding table creation fences");
            device.destroy_fence(raygen_fence, None);
            device.destroy_fence(miss_fence, None);
            device.destroy_fence(hit_fence, None);
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

    let (camera_lens_buffer, fence) = utils::buffers::new_data_buffer(
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
    unsafe {
        device
            .wait_for_fences(&[fence], true, FIVE_SECONDS_IN_NANOSECONDS)
            .expect("Failed to wait for camera lens buffer creation fence");
        device.destroy_fence(fence, None);
    }

    // Clean up the staging buffer.
    staging.destroy(device, allocator);

    camera_lens_buffer
}

/// A helper type for managing the descriptor sets for the ray tracing pipeline.
struct Descriptors {
    pub descriptor_pool: ash::vk::DescriptorPool,
    pub acceleration_set: ash::vk::DescriptorSet,
    pub image_sets: Vec<ash::vk::DescriptorSet>,
    pub camera_lens_set: ash::vk::DescriptorSet,
    pub camera_lens_buffer: utils::buffers::BufferAllocation,
}
impl Descriptors {
    /// Destroy the descriptor pool and descriptor sets.
    pub fn destroy(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }

        // Destroy the camera lens uniform buffer.
        self.camera_lens_buffer.destroy(device, allocator);
    }
}

fn create_projection_matrix(extent: ash::vk::Extent2D) -> glm::Mat4 {
    glm::Mat4::new_perspective(
        extent.width as f32 / extent.height as f32,
        std::f32::consts::FRAC_PI_2 * (7.5 / 9.), // 75 degrees.
        0.1,
        512.,
    )
}

/// Create the descriptor sets to store the acceleration structure storage images.
fn create_descriptor_sets(
    device: &ash::Device,
    output_images: &[ash::vk::ImageView],
    descriptor_set_layouts: &DescriptorSetLayouts,
    top_acceleration: &utils::ray_tracing::AccelerationStructure,
    camera_lens_buffer: utils::buffers::BufferAllocation,
) -> Descriptors {
    let image_count = output_images.len() as u32;

    // Create the descriptor pool for the ray tracer.
    // TODO: Allow for texture and mesh descriptor sets.
    // TODO: Consider whether multiple top-level acceleration structures are needed.
    //       It could facilitate updating the scene in-flight, but I don't know for certain.
    // TODO: Calculate `max_sets` based on the `DescriptorSetLayouts`.
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(2 + image_count)
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
                ]),
            None,
        )
    }
    .expect("Failed to create ray tracer descriptor pool");

    // Allocate the acceleration descriptor set.
    let acceleration_descriptor_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layouts.acceleration));
    let acceleration_set =
        unsafe { device.allocate_descriptor_sets(&acceleration_descriptor_info) }
            .expect("Failed to allocate acceleration descriptor set for the ray tracer")[0];

    // TODO: Reduce the number of `allocate_descriptor_sets` calls.
    // To avoid updating the output image descriptor set for each frame, we allocate one per frame.
    let image_descriptor_info = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layouts.output_image));
    let image_sets = output_images
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

    // Set the update data for each output image.
    let image_info = output_images
        .iter()
        .map(|image_view| {
            ash::vk::DescriptorImageInfo::default()
                .image_view(*image_view)
                .image_layout(ash::vk::ImageLayout::GENERAL)
        })
        .collect::<Vec<_>>();
    let descriptor_set_writes = std::iter::once(acceleration_write)
        .chain(std::iter::once(camera_lens_write))
        .chain(
            image_info
                .iter()
                .zip(image_sets.iter())
                .map(|(image_info, image_set)| {
                    // Update the bindings for each output image.
                    ash::vk::WriteDescriptorSet::default()
                        .dst_set(*image_set)
                        .dst_binding(0)
                        .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(image_info))
                }),
        )
        .collect::<Vec<_>>();

    // Update the descriptor sets with the new bindings.
    unsafe {
        device.update_descriptor_sets(&descriptor_set_writes, &[]);
    }

    Descriptors {
        descriptor_pool,
        acceleration_set,
        image_sets,
        camera_lens_set,
        camera_lens_buffer,
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
                matrix: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            },
        );
        let mesh_instance_1 = create_mesh_instance(
            bottom_acceleration.device_address(),
            ash::vk::TransformMatrixKHR {
                matrix: [0., 0.2, 0., 60., -0.2, 0., 0., -60., 0., 0., 0.2, 60.],
            },
        );
        let mesh_instance_2 = create_mesh_instance(
            bottom_acceleration.device_address(),
            ash::vk::TransformMatrixKHR {
                matrix: [0., 0., 0.2, -60., 0., 0.2, 0., -60., -0.2, 0., 0., -60.],
            },
        );

        let top_acceleration = create_top_level_acceleration_structure(
            device,
            acceleration_device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            &[mesh_instance, mesh_instance_1, mesh_instance_2],
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

        let camera_lens = create_lens_buffer(
            device,
            pageable_device_local_memory,
            allocator,
            command_pool,
            queue,
            extent,
        );

        let descriptors = create_descriptor_sets(
            device,
            output_images,
            &pipeline.descriptor_set_layouts,
            &top_acceleration,
            camera_lens,
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

        // TODO: Use `Descriptors` to generate this array.
        let descriptor_sets = [
            self.descriptors.acceleration_set,
            self.descriptors.image_sets[image_index as usize],
            self.descriptors.camera_lens_set,
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
                ash::vk::ShaderStageFlags::RAYGEN_KHR /*TODO: | ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR */,
                0,
                utils::data_byte_slice(push_constants)
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
    pub fn recreate_view_sets(
        &mut self,
        device: &ash::Device,
        pageable_device_local_memory: Option<&ash::ext::pageable_device_local_memory::Device>,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
        output_images: &[ash::vk::ImageView],
        extent: ash::vk::Extent2D,
    ) {
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
        let camera_update_fence = utils::buffers::update_device_local(
            device,
            allocator,
            command_pool,
            queue,
            &self.descriptors.camera_lens_buffer,
            utils::data_byte_slice(&projection_inverse),
            &mut staging,
            None,
        )
        .expect("Failed to update camera lens buffer for the ray tracing demo");

        // Wait for the fence to signal that the buffer is ready, then delete the fence.
        unsafe {
            device
                .wait_for_fences(&[camera_update_fence], true, FIVE_SECONDS_IN_NANOSECONDS)
                .expect("Failed to wait for camera lens buffer update fence");
            device.destroy_fence(camera_update_fence, None);
        }

        // TODO: Figure out how to have this passed back to the caller for destruction, along with
        //       the fence. It's not necessary to wait for the copy to complete now.
        staging.destroy(device, allocator);

        // Create the descriptor write objects.
        let descriptor_set_writes = image_info
            .iter()
            .zip(self.descriptors.image_sets.iter())
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
    }
}
