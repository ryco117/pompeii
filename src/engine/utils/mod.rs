use std::{
    collections::HashSet,
    ffi::{c_void, CStr},
};

use smallvec::SmallVec;
use strum::EnumCount as _;

pub mod buffers;
pub mod fxaa_pass;
pub mod ray_tracing;
pub mod swapchain;
pub mod textures;

// Expose the `swapchain` module for simplicity.
pub use swapchain::*;

/// Store the SPIR-V representation of the shaders in the binary.
pub mod shaders {
    /// The default entry point for shaders is the `main` function.
    pub const ENTRY_POINT_MAIN: &std::ffi::CStr = c"main";

    /// Shader for texture-mapping the entire screen. Useful for post-processing and fullscreen effects.
    pub const FULLSCREEN_VERTEX: &[u32] =
        inline_spirv::include_spirv!("src/shaders/fullscreen_vert.glsl", vert, glsl);
}

/// Target Vulkan API version 1.3 for compatibility with the latest Vulkan features and **reduced fragmentation of extension support**.
const VULKAN_API_VERSION: u32 = ash::vk::API_VERSION_1_3;

/// Set a sane value for the maximum expected number of queue families.
/// A heap allocation is required if the number of queue families exceeds this value.
pub const EXPECTED_MAX_QUEUE_FAMILIES: usize = 8;

/// Set a sane value for the maximum expected number of instance extensions.
/// A heap allocation is required if the number of instance extensions exceeds this value.
pub const EXPECTED_MAX_ENABLED_INSTANCE_EXTENSIONS: usize = 8;

/// Set a sane value for the maximum expected number of physical devices.
/// A heap allocation is required if the number of physical devices exceeds this value.
pub const EXPECTED_MAX_VULKAN_PHYSICAL_DEVICES: usize = 4;

/// Sane maximum number of frames-in-flight before certain heap allocations are required.
pub const EXPECTED_MAX_FRAMES_IN_FLIGHT: usize = 4;

/// The number of nanoseconds in five seconds. Used for sane timeouts on synchronization objects.
pub const FIVE_SECONDS_IN_NANOSECONDS: u64 = 5_000_000_000;

/// Helper to check if a list of available extensions contains a specific extension name.
pub fn extensions_list_contains(list: &[ash::vk::ExtensionProperties], ext: &CStr) -> bool {
    list.iter().any(|p| {
        p.extension_name_as_c_str()
            .expect("Extension name received from Vulkan is not a valid C-string")
            == ext
    })
}

/// Helper for returning a fence with the necessary cleanup.
pub struct CleanableFence {
    pub fence: ash::vk::Fence,
    pub cleanup: Option<Box<dyn FnOnce(&ash::Device, &mut gpu_allocator::vulkan::Allocator)>>,
}
impl CleanableFence {
    /// Create a new `CleanableFence` with the specified fence and cleanup function.
    /// # Safety
    /// Do not destroy the fence from within the `cleanup` function. This will be done automatically.
    pub fn new(
        fence: ash::vk::Fence,
        cleanup: Option<Box<dyn FnOnce(&ash::Device, &mut gpu_allocator::vulkan::Allocator)>>,
    ) -> Self {
        Self { fence, cleanup }
    }

    /// Destroy the fence and invoke the `cleanup` function before dropping the instance.
    pub fn cleanup(self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        unsafe {
            device.destroy_fence(self.fence, None);
        }

        if let Some(cleanup) = self.cleanup {
            cleanup(device, allocator);
        }
    }
}

/// A minimal helper for converting a Rust slice `&[T]` to a byte slice over the same memory.
/// # Safety
/// This function uses `unsafe` code to create a slice from a casted pointer and the `std::mem::size_of_val` bytes of memory.
pub fn data_slice_byte_slice<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(data).cast(), std::mem::size_of_val(data))
    }
}

/// A minimal helper for converting a Rust reference `&T` to a byte slice over the same memory.
/// # Safety
/// This function uses `unsafe` code to create a slice from a casted pointer and the `std::mem::size_of_val` bytes of memory.
pub fn data_byte_slice<T>(data: &T) -> &[u8] {
    data_slice_byte_slice(std::slice::from_ref(data))
}

/// A helper to round up a size to the nearest multiple of an alignment.
/// # Safety
/// The alignment must be a power of two.
pub fn aligned_size<T>(size: T, alignment: T) -> T
where
    T: num_traits::PrimInt + num_traits::Unsigned,
{
    #[cfg(debug_assertions)]
    assert!(
        alignment.count_ones() == 1,
        "Alignment must be a power of two"
    );

    // Get the value of one for the type `T`.
    let one = T::one();

    // Efficient trick to round up to an alignment when the alignment is a power of two.
    // Works as follows:
    // 1 Adds the alignment minus one to the size.
    //  * This ensures that the alignment bit is increased if and only if size is not already aligned.
    // 2 Bitwise AND with the negation of the alignment minus one.
    //  * This ensures that the bits less than our alignment are zero, enforcing alignment.
    (size + alignment - one) & !(alignment - one)
}

#[cfg(debug_assertions)]
#[derive(strum::Display)]
#[strum(serialize_all = "snake_case")]
/// Let helper functions know what kind of extension is being operated on.
pub enum ExtensionType {
    Instance,
    Device,
}

/// Helper to easily add all of the `dependent_extensions` to the `total` list if they are all available.
/// Returns `true` if all extensions are available and added to the `total` list.
/// Otherwise, returns `false` and the `total` list is unchanged.
pub fn extend_if_all_extensions_are_contained(
    available_extensions: &[ash::vk::ExtensionProperties],
    dependent_extensions: &[&CStr],
    total: &mut Vec<*const i8>,
    #[cfg(debug_assertions)] extension_type: ExtensionType,
) -> bool {
    if dependent_extensions
        .iter()
        .all(|ext| extensions_list_contains(available_extensions, ext))
    {
        total.extend(dependent_extensions.iter().map(|ext| {
            #[cfg(debug_assertions)]
            println!(
                "INFO: Enabling {} {extension_type} extension",
                ext.to_string_lossy()
            );

            ext.as_ptr()
        }));
        true
    } else {
        false
    }
}

/// The possible errors that may occur when creating a `VulkanCore`.
#[derive(Debug)]
pub enum VulkanCoreError {
    Loading(ash::LoadingError),
    MissingExtension(String),
    MissingLayer(String),
}

/// The main Vulkan library interface. Contains the entry to the Vulkan library and an instance for this app.
pub struct VulkanCore {
    pub version: u32,
    pub api: ash::Entry,
    pub instance: ash::Instance,
    pub surface_instance: Option<ash::khr::surface::Instance>,
    enabled_instance_extensions: HashSet<&'static CStr>,
}

impl VulkanCore {
    /// Create a new `VulkanCore` with the specified instance extensions.
    pub fn new(
        required_extensions: &[&'static CStr],
        optional_extensions: &[&'static CStr],
    ) -> Result<Self, VulkanCoreError> {
        // Attempt to dynamically load the Vulkan API from platform-specific shared libraries.
        let vulkan_api = unsafe { ash::Entry::load().map_err(VulkanCoreError::Loading)? };

        // Determine which extensions are available at runtime.
        let available_extensions = unsafe {
            vulkan_api
                .enumerate_instance_extension_properties(None)
                .expect("Unable to enumerate available Vulkan extensions")
        };

        #[cfg(debug_assertions)]
        println!("INFO: Available instance extensions: {available_extensions:?}\n");

        // Check that all of the required extensions are available.
        if let Some(missing) = required_extensions
            .iter()
            .find(|ext| !extensions_list_contains(&available_extensions, ext))
        {
            return Err(VulkanCoreError::MissingExtension(
                missing.to_string_lossy().into_owned(),
            ));
        }

        // Track which extensions are being enabled and convert those extensions to a list of pointers.
        let mut enabled_instance_extensions = std::collections::HashSet::new();
        let mut extension_name_pointers: Vec<*const i8> = required_extensions
            .iter()
            .map(|&e| {
                // Add the required extensions to the enabled set.
                enabled_instance_extensions.insert(e);

                // Map the extension to a pointer representation.
                e.as_ptr()
            })
            .collect();

        // Add all of the optional extensions to our creation set if they are available.
        for extension in optional_extensions {
            if extensions_list_contains(&available_extensions, extension) {
                #[cfg(debug_assertions)]
                println!("INFO: Enabling {extension:?} instance extension");

                enabled_instance_extensions.insert(extension);
                extension_name_pointers.push(extension.as_ptr());
            }
        }

        // Optionally, enable `VK_EXT_swapchain_colorspace` if is is available and the dependent `VK_KHR_surface` is requested.
        let requiring_khr_surface = required_extensions.contains(&ash::khr::surface::NAME);
        if requiring_khr_surface {
            extend_if_all_extensions_are_contained(
                &available_extensions,
                &[ash::ext::swapchain_colorspace::NAME],
                &mut extension_name_pointers,
                #[cfg(debug_assertions)]
                ExtensionType::Instance,
            );
        }

        // Optionally, enable `VK_KHR_get_surface_capabilities2` and `VK_EXT_surface_maintenance1` if they are available and the dependent `VK_KHR_surface` is requested.
        if requiring_khr_surface {
            extend_if_all_extensions_are_contained(
                &available_extensions,
                &[
                    ash::khr::get_surface_capabilities2::NAME,
                    ash::ext::surface_maintenance1::NAME,
                ],
                &mut extension_name_pointers,
                #[cfg(debug_assertions)]
                ExtensionType::Instance,
            );
        }

        // Add the debug utility extension if in debug mode.
        #[cfg(debug_assertions)]
        extend_if_all_extensions_are_contained(
            &available_extensions,
            &[ash::ext::debug_utils::NAME],
            &mut extension_name_pointers,
            ExtensionType::Instance,
        );

        // Enable validation layers when using a debug build.
        #[cfg(debug_assertions)]
        let layer_names = {
            // Enable the main validation layer from the Khronos Group when using a debug build.
            const DEBUG_LAYERS: [*const i8; 1] = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

            // Check which layers are available at runtime.
            let available_layers = unsafe {
                vulkan_api
                    .enumerate_instance_layer_properties()
                    .expect("Unable to enumerate available Vulkan layers")
            };
            println!("INFO: Available layers: {available_layers:?}\n");

            // Check that all the desired debug layers are available.
            if let Some(missing) = DEBUG_LAYERS.iter().find_map(|&layer_ptr| {
                let layer_cstr = unsafe { CStr::from_ptr(layer_ptr) };
                let layer_exists = available_layers.iter().any(|a| {
                    a.layer_name_as_c_str()
                        .expect("Available layer name is not a valid C-string")
                        == layer_cstr
                });

                if layer_exists {
                    // This layer is not missing from the available layers.
                    None
                } else {
                    // This layer is missing, return it as a `String` to the `find_map`.
                    Some(layer_cstr.to_string_lossy().into_owned())
                }
            }) {
                return Err(VulkanCoreError::MissingLayer(missing));
            }

            DEBUG_LAYERS
        };
        // Disable validation layers when using a release build.
        #[cfg(not(debug_assertions))]
        let layer_names = [];

        // Create a Vulkan instance with the given extensions and layers.
        let vulkan_instance = {
            let application_info = ash::vk::ApplicationInfo {
                p_application_name: c"Pompeii".as_ptr().cast(),
                application_version: ash::vk::make_api_version(0, 0, 1, 0),
                p_engine_name: c"Pompeii".as_ptr().cast(),
                engine_version: ash::vk::make_api_version(0, 0, 1, 0),
                api_version: VULKAN_API_VERSION,
                ..Default::default()
            };
            let instance_info = {
                ash::vk::InstanceCreateInfo {
                    p_application_info: &application_info,
                    enabled_layer_count: layer_names.len() as u32,
                    pp_enabled_layer_names: layer_names.as_ptr(),
                    enabled_extension_count: extension_name_pointers.len() as u32,
                    pp_enabled_extension_names: extension_name_pointers.as_ptr(),
                    ..Default::default()
                }
            };
            unsafe {
                vulkan_api
                    .create_instance(&instance_info, None)
                    .expect("Unable to create Vulkan instance")
            }
        };

        let surface_instance = if requiring_khr_surface {
            Some(ash::khr::surface::Instance::new(
                &vulkan_api,
                &vulkan_instance,
            ))
        } else {
            None
        };
        Ok(Self {
            version: VULKAN_API_VERSION,
            api: vulkan_api,
            instance: vulkan_instance,
            surface_instance,
            enabled_instance_extensions,
        })
    }

    /// Check if the Vulkan instance was created with a specific extension enabled.
    pub fn enabled_instance_extension(&self, ext: &CStr) -> bool {
        self.enabled_instance_extensions.contains(ext)
    }
}

/// The Vulkan physical device features which this engine may utilize. This struct is used to query and report enabled features.
#[derive(Clone, Copy, Debug, Default)]
pub struct EnginePhysicalDeviceFeatures {
    /// Corresponds to `VkPhysicalDeviceAccelerationStructureFeaturesKHR`.
    pub acceleration_structure: ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,

    /// Corresponds to `VkPhysicalDeviceBufferDeviceAddressFeatures`.
    pub buffer_device_address: ash::vk::PhysicalDeviceBufferDeviceAddressFeatures<'static>,

    /// Corresponds to `VkPhysicalDeviceDescriptorIndexingFeatures`.
    pub descriptor_indexing: ash::vk::PhysicalDeviceDescriptorIndexingFeatures<'static>,

    /// Corresponds to `VkPhysicalDeviceDynamicRenderingFeatures`.
    pub dynamic_rendering: ash::vk::PhysicalDeviceDynamicRenderingFeatures<'static>,

    /// Corresponds to `VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT`.
    pub pageable_device_local_memory:
        ash::vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT<'static>,

    /// Corresponds to `VkPhysicalDeviceRayQueryFeaturesKHR`.
    pub ray_query: ash::vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,

    /// Corresponds to `VkPhysicalDeviceRayTracingPipelineFeaturesKHR`.
    pub ray_tracing: ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,

    /// Corresponds to `VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR`.
    pub ray_tracing_maintenance: ash::vk::PhysicalDeviceRayTracingMaintenance1FeaturesKHR<'static>,

    /// Corresponds to `VkPhysicalDeviceSynchronization2Features`.
    pub synchronization2: ash::vk::PhysicalDeviceSynchronization2Features<'static>,
}
impl EnginePhysicalDeviceFeatures {
    pub fn all() -> Self {
        let mut all = Self::default();
        all.acceleration_structure.acceleration_structure = ash::vk::TRUE;
        all.buffer_device_address.buffer_device_address = ash::vk::TRUE;
        all.dynamic_rendering.dynamic_rendering = ash::vk::TRUE;
        all.pageable_device_local_memory
            .pageable_device_local_memory = ash::vk::TRUE;
        all.ray_query.ray_query = ash::vk::TRUE;
        all.ray_tracing.ray_tracing_pipeline = ash::vk::TRUE;
        all.ray_tracing_maintenance.ray_tracing_maintenance1 = ash::vk::TRUE;
        all.synchronization2.synchronization2 = ash::vk::TRUE;
        all
    }

    /// A helper to verify that this feature-set contains every enabled (i.e., `true`) feature in the provided mask.
    /// # Note
    /// This doesn't contain checking for sub-properties of a feature. For example, `descriptor_indexing` is not checked because it is only sub-properties.
    pub fn contains_mask(self, mask: &EnginePhysicalDeviceFeatures) -> bool {
        (!mask.acceleration_structure() || self.acceleration_structure())
            && (!mask.buffer_device_address() || self.buffer_device_address())
            && (!mask.dynamic_rendering() || self.dynamic_rendering())
            && (!mask.pageable_device_local_memory() || self.pageable_device_local_memory())
            && (!mask.ray_query() || self.ray_query())
            && (!mask.ray_tracing() || self.ray_tracing())
            && (!mask.ray_tracing_maintenance() || self.ray_tracing_maintenance())
            && (!mask.synchronization2() || self.synchronization2())
    }

    /// Query a physical device for support of a given set of features.
    /// If a feature has an element which indicates the state of the entire feature,
    /// then that value is used to determine whether the feature should be included in this query or not.
    /// # Note
    /// The `p_next` pointers will be ignored by this function and set to `null` before returning.
    pub fn query_physical_feature_support(
        &mut self,
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
    ) -> ash::vk::PhysicalDeviceFeatures {
        // Build the feature chain from the provided features.
        let mut feature_chain = ash::vk::PhysicalDeviceFeatures2::default();

        // Ensure there are no circular references in the feature chain.
        self.clear_pointers();

        if self.acceleration_structure() {
            feature_chain = feature_chain.push_next(&mut self.acceleration_structure);
        }
        if self.buffer_device_address.buffer_device_address == ash::vk::TRUE {
            feature_chain = feature_chain.push_next(&mut self.buffer_device_address);
        }
        feature_chain = feature_chain.push_next(&mut self.descriptor_indexing);
        if self.dynamic_rendering.dynamic_rendering == ash::vk::TRUE {
            feature_chain = feature_chain.push_next(&mut self.dynamic_rendering);
        }
        if self.synchronization2.synchronization2 == ash::vk::TRUE {
            feature_chain = feature_chain.push_next(&mut self.synchronization2);
        }
        if self
            .pageable_device_local_memory
            .pageable_device_local_memory
            == ash::vk::TRUE
        {
            feature_chain = feature_chain.push_next(&mut self.pageable_device_local_memory);
        }
        if self.ray_query.ray_query == ash::vk::TRUE {
            feature_chain = feature_chain.push_next(&mut self.ray_query);
        }
        if self.ray_tracing.ray_tracing_pipeline == ash::vk::TRUE {
            feature_chain = feature_chain.push_next(&mut self.ray_tracing);
        }
        if self.ray_tracing_maintenance.ray_tracing_maintenance1 == ash::vk::TRUE {
            feature_chain = feature_chain.push_next(&mut self.ray_tracing_maintenance);
        }

        let features = {
            // Query the physical device for the selected features.
            unsafe { instance.get_physical_device_features2(physical_device, &mut feature_chain) };
            feature_chain.features
        };

        // Don't confuse the caller with pointers to features.
        self.clear_pointers();

        // Return the features described in the default structure.
        features
    }

    /// Set the required toggles to enable or disable the Vulkan ray-tracing (RTX) feature.
    /// # Note
    /// This function should only be called with `enable==true` if the internal structs are intended to be used to
    /// query the physical device for support, and not as the truth of a device's support yet.
    pub fn set_ray_tracing(&mut self, enable: bool) {
        let enable = if enable {
            ash::vk::TRUE
        } else {
            ash::vk::FALSE
        };
        self.acceleration_structure.acceleration_structure = enable;
        self.ray_query.ray_query = enable;
        self.ray_tracing.ray_tracing_pipeline = enable;
        self.ray_tracing_maintenance.ray_tracing_maintenance1 = enable;
    }

    // Getters for features with a single state representing their support.
    pub fn acceleration_structure(&self) -> bool {
        self.acceleration_structure.acceleration_structure == ash::vk::TRUE
    }
    pub fn buffer_device_address(&self) -> bool {
        self.buffer_device_address.buffer_device_address == ash::vk::TRUE
    }
    pub fn dynamic_rendering(&self) -> bool {
        self.dynamic_rendering.dynamic_rendering == ash::vk::TRUE
    }
    pub fn pageable_device_local_memory(&self) -> bool {
        self.pageable_device_local_memory
            .pageable_device_local_memory
            == ash::vk::TRUE
    }
    pub fn ray_query(&self) -> bool {
        self.ray_query.ray_query == ash::vk::TRUE
    }
    pub fn ray_tracing(&self) -> bool {
        self.ray_tracing.ray_tracing_pipeline == ash::vk::TRUE
    }
    pub fn ray_tracing_maintenance(&self) -> bool {
        self.ray_tracing_maintenance.ray_tracing_maintenance1 == ash::vk::TRUE
    }
    pub fn synchronization2(&self) -> bool {
        self.synchronization2.synchronization2 == ash::vk::TRUE
    }

    /// Clear all feature pointers to `NULL` to avoid circular references.
    pub fn clear_pointers(&mut self) {
        self.acceleration_structure.p_next = std::ptr::null_mut::<c_void>();
        self.buffer_device_address.p_next = std::ptr::null_mut::<c_void>();
        self.descriptor_indexing.p_next = std::ptr::null_mut::<c_void>();
        self.dynamic_rendering.p_next = std::ptr::null_mut::<c_void>();
        self.pageable_device_local_memory.p_next = std::ptr::null_mut::<c_void>();
        self.ray_query.p_next = std::ptr::null_mut::<c_void>();
        self.ray_tracing.p_next = std::ptr::null_mut::<c_void>();
        self.ray_tracing_maintenance.p_next = std::ptr::null_mut::<c_void>();
        self.synchronization2.p_next = std::ptr::null_mut::<c_void>();
    }
}

/// Get all physical devices that support the minimum Vulkan API and the required extensions.
/// Sort them by their likelihood of being the desired device.
pub fn get_sorted_physical_devices(
    instance: &ash::Instance,
    minimum_version: u32,
    required_extensions: &[*const i8],
    required_features: &EnginePhysicalDeviceFeatures,
) -> SmallVec<
    [(
        ash::vk::PhysicalDevice,
        ash::vk::PhysicalDeviceProperties,
        EnginePhysicalDeviceFeatures,
    ); EXPECTED_MAX_VULKAN_PHYSICAL_DEVICES],
> {
    /// A helper for scoring a device's dedication to graphics processing.
    fn score_device_type(device_type: ash::vk::PhysicalDeviceType) -> u8 {
        match device_type {
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            ash::vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            ash::vk::PhysicalDeviceType::CPU => 1,
            _ => 0,
        }
    }

    // Get all physical devices that support Vulkan 1.3.
    // TODO: Investigate if `enumerate_physical_devices` can be replaced with something I can pass a `SmallVec` ref to.
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Unable to enumerate physical devices")
    };

    let mut physical_devices = physical_devices
        .into_iter()
        .filter_map(|device| {
            // Query basic properties of the physical device.
            let properties = unsafe { instance.get_physical_device_properties(device) };

            // Query for the basic extensions available to this physical device.
            let extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Unable to enumerate device extensions")
            };

            // Ensure every device has their properties printed in debug mode. Called before filtering starts.
            #[cfg(debug_assertions)]
            println!("INFO: Physical device {device:?}: {properties:?}\nPhysical device available extensions: {extensions:?}\n");

            // Ensure the device supports Vulkan 1.3.
            if properties.api_version < minimum_version {
                #[cfg(debug_assertions)]
                println!("INFO: Physical device {device:?} does not support a sufficiently high Vulkan version");
                return None;
            }

            // Ensure the device supports all required features.
            let mut copy_features = EnginePhysicalDeviceFeatures::all();
            let _ = copy_features.query_physical_feature_support(instance, device);
            if !copy_features.contains_mask(required_features) {
                #[cfg(debug_assertions)]
                println!("INFO: Physical device {device:?}: does not support all the required features {required_features:?}: Actual features {copy_features:?}");
                return None;
            }

            // Ensure the device supports all required extensions.
            if required_extensions.iter().all(|&req| {
                let req = unsafe { CStr::from_ptr(req) };
                let exists = extensions_list_contains(&extensions, req);

                #[cfg(debug_assertions)]
                if !exists {
                    println!("INFO: Physical device {device:?} does not support required extension '{req:?}'");
                }
                exists
            }) {
                Some((device, properties, copy_features))
            } else {
                None
            }
        })
        .collect::<SmallVec<[(ash::vk::PhysicalDevice, ash::vk::PhysicalDeviceProperties, EnginePhysicalDeviceFeatures); 4]>>();

    #[cfg(debug_assertions)]
    if physical_devices.spilled() {
        println!(
            "INFO: Physical devices list has spilled over to the heap. Device count {} greater than inline size {}",
            physical_devices.len(),
            physical_devices.inline_size(),
        );
    }

    // Sort the physical devices by the device type with preference for GPU's, then descending by graphics dedication.
    physical_devices.sort_by(|(_, a, _), (_, b, _)| {
        // Sorting order is reversed (`b.cmp(a)`) to sort the highest scoring device first.
        let device_type_cmp =
            score_device_type(b.device_type).cmp(&score_device_type(a.device_type));

        // If device types are equivalent, then sort by the maximum push constants size.
        if device_type_cmp == std::cmp::Ordering::Equal {
            b.limits
                .max_push_constants_size
                .cmp(&a.limits.max_push_constants_size)
        } else {
            device_type_cmp
        }
    });

    physical_devices
}

/// Available queue family types and their indices.
/// Also, a map of queue family indices to the number of queues that may be and are currently allocated.
pub struct QueueFamilies {
    pub graphics: Vec<u32>,
    pub compute: Vec<u32>,
    pub transfer: Vec<u32>,
    pub present: Vec<u32>,
    pub queue_families: Vec<ash::vk::QueueFamilyProperties>,
}

#[derive(strum::EnumCount)]
#[repr(usize)]
pub enum QueueType {
    Graphics,
    Compute,
    Present,
    Transfer,
}

/// Get the necessary queue family indices for a logical device capable of graphics, compute, and presentation.
pub fn get_queue_families(
    vulkan: &VulkanCore,
    physical_device: ash::vk::PhysicalDevice,
    surface: ash::vk::SurfaceKHR,
) -> QueueFamilies {
    // Get the list of available queue families for this device.
    let queue_families = unsafe {
        vulkan
            .instance
            .get_physical_device_queue_family_properties(physical_device)
    };

    #[cfg(debug_assertions)]
    println!("INFO: Queue families: {queue_families:?}\n");

    // Find the queue families for each desired queue type.
    // Use an array with indices to allow compile-time guarantees about the number of queue types.
    let mut type_indices = [const { Vec::<u32>::new() }; QueueType::COUNT];
    for (family_index, queue_family) in queue_families.iter().enumerate() {
        // Get as a present queue family.
        if let Some(surface_instance) = vulkan.surface_instance.as_ref() {
            if unsafe {
                surface_instance.get_physical_device_surface_support(
                    physical_device,
                    family_index as u32,
                    surface,
                )
            }
            .expect("Unable to check if 'present' is supported")
            {
                type_indices[QueueType::Present as usize].push(family_index as u32);
            }
        }

        // Get as a graphics queue family.
        if queue_family
            .queue_flags
            .contains(ash::vk::QueueFlags::GRAPHICS)
        {
            type_indices[QueueType::Graphics as usize].push(family_index as u32);
        }

        // Get as a compute queue family.
        if queue_family
            .queue_flags
            .contains(ash::vk::QueueFlags::COMPUTE)
        {
            type_indices[QueueType::Compute as usize].push(family_index as u32);
        }

        // Get as a transfer queue family.
        if queue_family
            .queue_flags
            .contains(ash::vk::QueueFlags::TRANSFER)
        {
            type_indices[QueueType::Transfer as usize].push(family_index as u32);
        }
    }

    let graphics = std::mem::take(&mut type_indices[QueueType::Graphics as usize]);
    let compute = std::mem::take(&mut type_indices[QueueType::Compute as usize]);
    let transfer = std::mem::take(&mut type_indices[QueueType::Transfer as usize]);
    let present = std::mem::take(&mut type_indices[QueueType::Present as usize]);

    QueueFamilies {
        graphics,
        compute,
        transfer,
        present,
        queue_families,
    }
}

/// Create a Vulkan logical device capable of graphics, compute, and presentation queues.
/// Returns the device and the queue family indices that were requested for use.
/// # Safety
/// The behavior is undefined if the physical device does not support all the requested device extensions or features.
pub fn new_device(
    vulkan: &VulkanCore,
    physical_device: ash::vk::PhysicalDevice,
    surface: ash::vk::SurfaceKHR,
    device_extensions: &[*const i8],
    feature_chain: Option<&mut dyn ash::vk::ExtendsDeviceCreateInfo>,
) -> (ash::Device, QueueFamilies) {
    // Get the necessary queue family indices for the logical device.
    let queue_families = get_queue_families(vulkan, physical_device, surface);

    // Aggregate queue family indices and count the number of queues each family may need allocated.
    let mut family_map = SmallVec::<[u32; EXPECTED_MAX_QUEUE_FAMILIES]>::from_elem(
        0,
        queue_families.queue_families.len(),
    );
    let all_family_types = [
        &queue_families.graphics,
        &queue_families.compute,
        &queue_families.present,
    ];
    for family in all_family_types {
        for index in family {
            family_map[*index as usize] += 1;
        }
    }

    // Allocate a vector of queue priorities capable of handling the largest queue count among all families.
    let priorities = vec![
        1.;
        queue_families
            .queue_families
            .iter()
            .fold(0, |acc, f| acc.max(f.queue_count as usize))
    ];

    // Print a message if the queue family map has spilled over to the heap.
    #[cfg(debug_assertions)]
    if family_map.spilled() {
        println!(
            "INFO: Queue family map has spilled over to the heap. Family count {} greater than inline size {}",
            queue_families.queue_families.len(),
            family_map.inline_size(),
        );
    }

    // Describe the queue families that will be used with the new logical device.
    let queue_info = family_map
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| {
            if count == 0 {
                return None;
            }

            // Limit to the number of available queues. Guaranteed to be non-zero.
            let queue_count = count.min(queue_families.queue_families[index].queue_count);
            let priorities = &priorities[..queue_count as usize];

            Some(ash::vk::DeviceQueueCreateInfo {
                queue_family_index: index as u32,
                queue_count,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            })
        })
        .collect::<Vec<_>>();

    // Create the logical device with the desired queue families and extensions.
    let device = unsafe {
        let mut device_info = ash::vk::DeviceCreateInfo {
            queue_create_info_count: queue_info.len() as u32,
            p_queue_create_infos: queue_info.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };

        // Optionally, add additional features to the device creation info.
        if let Some(feature_chain) = feature_chain {
            device_info = device_info.push_next(feature_chain);
        }

        vulkan
            .instance
            .create_device(physical_device, &device_info, None)
            .expect("Unable to create logical device")
    };

    (device, queue_families)
}

/// A helper type for understanding the context of a queue in Vulkan.
pub struct IndexedQueue {
    pub queue: ash::vk::Queue,
    pub family_index: u32,
    pub index: u32,
}
impl IndexedQueue {
    /// Get a queue from a logical device by its family and index.
    pub fn get(device: &ash::Device, family_index: u32, index: u32) -> Self {
        let queue = unsafe { device.get_device_queue(family_index, index) };
        Self {
            queue,
            family_index,
            index,
        }
    }
}

/// A helper for creating shader modules on a logical device.
pub fn create_shader_module(device: &ash::Device, code: &[u32]) -> ash::vk::ShaderModule {
    unsafe {
        device.create_shader_module(
            &ash::vk::ShaderModuleCreateInfo {
                code_size: code.len() << 2, // Expects the code size to be in bytes.
                p_code: code.as_ptr(),
                ..Default::default()
            },
            None,
        )
    }
    .expect("Unable to create shader module")
}

/// Check if the image format is for a depth buffer.
pub fn is_depth_format(format: ash::vk::Format) -> bool {
    use ash::vk::Format;
    matches!(
        format,
        Format::D16_UNORM
            | Format::D16_UNORM_S8_UINT
            | Format::D24_UNORM_S8_UINT
            | Format::D32_SFLOAT
            | Format::D32_SFLOAT_S8_UINT
            | Format::X8_D24_UNORM_PACK32
    )
}

/// Check if the image format is for a stencil buffer.
pub fn is_stencil_format(format: ash::vk::Format) -> bool {
    use ash::vk::Format;
    matches!(
        format,
        Format::S8_UINT
            | Format::D16_UNORM_S8_UINT
            | Format::D24_UNORM_S8_UINT
            | Format::D32_SFLOAT_S8_UINT
    )
}

/// Helper for creating a new image allocated by a `gpu_allocator::vulkan::Allocator`.
pub fn create_image(
    device: &ash::Device,
    memory_allocator: &mut gpu_allocator::vulkan::Allocator,
    image_create_info: &ash::vk::ImageCreateInfo,
    image_name: &str,
) -> (ash::vk::Image, gpu_allocator::vulkan::Allocation) {
    let image = unsafe { device.create_image(image_create_info, None) }
        .expect("Unable to create new image handle");

    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = memory_allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: image_name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(image),
        })
        .expect("Unable to allocate memory for new image");
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }
        .expect("Unable to bind memory to new image");

    (image, allocation)
}

/// Create a new image view for an existing Vulkan image with a specified format and MIP level.
pub fn create_image_view(
    device: &ash::Device,
    image: ash::vk::Image,
    format: ash::vk::Format,
    mip_levels: u32,
) -> ash::vk::ImageView {
    // Determine what kind of image view to create based on the format.
    let aspect_mask = if is_depth_format(format) {
        ash::vk::ImageAspectFlags::DEPTH
    } else if is_stencil_format(format) {
        ash::vk::ImageAspectFlags::STENCIL
    } else {
        ash::vk::ImageAspectFlags::COLOR
    };

    // Create the image view with the specified parameters.
    // NOTE: More parameters will be needed when supporting XR and 3D images.
    let view_info = ash::vk::ImageViewCreateInfo {
        image,
        view_type: ash::vk::ImageViewType::TYPE_2D, // 2D image. Use 2D array for multi-view/XR.
        format,
        components: ash::vk::ComponentMapping::default(),
        subresource_range: ash::vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: ash::vk::REMAINING_ARRAY_LAYERS, // In case of 3D images, `VK_REMAINING_ARRAY_LAYERS` will consider all remaining layers.
        },
        ..Default::default()
    };
    unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("Unable to create image view")
    }
}

/// A multi-sample anti-aliasing configuration, including the sample count and managed image resources.
struct MultiSampleAntiAliasing {
    pub samples: ash::vk::SampleCountFlags,
    pub images: Vec<(ash::vk::Image, gpu_allocator::vulkan::Allocation)>,
    pub image_views: Vec<ash::vk::ImageView>,
}

/// Query the physical device for the supported sample count for color images.
/// Returns `Some(n)` with the `ImageCreateInfo` for the single highest supported multi-sample count (i.e., `n > 1`) if found, else `None`.
pub fn query_multisample_support(
    vulkan: &VulkanCore,
    physical_device: ash::vk::PhysicalDevice,
    samples: ash::vk::SampleCountFlags,
    format: ash::vk::Format,
    extent: ash::vk::Extent2D,
    mip_levels: u32,
    usage: ash::vk::ImageUsageFlags,
) -> Option<ash::vk::ImageCreateInfo> {
    // If only checking for the single-sampling, return early.
    if samples.is_empty() || samples == ash::vk::SampleCountFlags::TYPE_1 {
        return None;
    }

    // Check if the physical device supports the requested sample count.
    let mut image_create = ash::vk::ImageCreateInfo {
        image_type: ash::vk::ImageType::TYPE_2D,
        format,
        extent: ash::vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
        mip_levels,
        array_layers: 1,
        samples,
        tiling: ash::vk::ImageTiling::OPTIMAL,
        usage,
        ..Default::default()
    };
    let limits = unsafe {
        vulkan
            .instance
            .get_physical_device_image_format_properties(
                physical_device,
                format,
                image_create.image_type,
                image_create.tiling,
                image_create.usage,
                image_create.flags,
            )
            .expect("Unable to get image format properties")
    };

    // Ensure that the physical device supports at least one color sample per pixel.
    #[cfg(debug_assertions)]
    {
        assert!(
            !limits.sample_counts.is_empty(),
            "Physical device does not support any color sample count"
        );
        println!(
            "INFO: Supported sample counts: {:?}\n",
            limits.sample_counts
        );
    }

    if limits.sample_counts.intersects(samples) {
        // Get the highest requested sample count.
        let mut specific_sample_count =
            ash::vk::SampleCountFlags::from_raw(1 << samples.as_raw().ilog2());

        // Find the highest supported sample count that is also requested.
        while !(specific_sample_count.is_empty()
            || limits.sample_counts.contains(specific_sample_count)
                && samples.contains(specific_sample_count))
        {
            specific_sample_count =
                ash::vk::SampleCountFlags::from_raw(specific_sample_count.as_raw() >> 1);
        }

        #[cfg(debug_assertions)]
        assert!(!specific_sample_count.is_empty(), "Physical device does not support any of the requested color sample count(s) {samples:?} after claiming it did");

        // Update the image create info with the supported sample count.
        image_create.samples = specific_sample_count;
        Some(image_create)
    } else {
        #[cfg(debug_assertions)]
        println!(
            "WARN: Physical device does not support any of the requested color sample count(s) {samples:?}\n"
        );

        None
    }
}

/// A helper to get a queue family index that supports the required queue flags and prefers
/// sharing the queue family with the specified flags, and not sharing with the rest.
pub fn get_queue_family_index(
    required_capabilities: ash::vk::QueueFlags,
    prefer_sharing_capabilities: ash::vk::QueueFlags,
    queue_families: &QueueFamilies,
) -> u32 {
    queue_families
        .queue_families
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            if f.queue_flags.contains(required_capabilities) {
                #[allow(clippy::cast_possible_wrap)]
                Some((
                    i as u32,
                    (f.queue_flags & prefer_sharing_capabilities)
                        .as_raw()
                        .count_ones()
                        .rotate_left(1) as i32
                        - (f.queue_flags & !prefer_sharing_capabilities)
                            .as_raw()
                            .count_ones() as i32,
                ))
            } else {
                None
            }
        })
        .max_by_key(|(_, score)| *score)
        .expect("Unable to find a queue family with the required flags")
        .0
}
