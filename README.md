# Pompeii

### Imposed restrictions
- (At least) Vulkan API version 1.3 must be used. This version greatly reduced the extension/feature fragmentation in the ecosystem and makes the engine both concise and portable.
- The chosen physical device must support the following device features. These features are common in modern hardware and allow for better primitives and API usage.
  * `VkPhysicalDeviceBufferDeviceAddressFeatures`
    - This is the least commonly supported, but enables very powerful memory access models from within shaders.
  * `VkPhysicalDeviceDynamicRenderingFeatures`
  * `VkPhysicalDeviceSynchronization2Features`
- Ray tracing support is conditioned on device support for the `VK_KHR_ray_tracing_maintenance1` extension so that the optimal synchronization flags can be used.

### Inspirations
In developing this playground I have seeked best-practices from a handful of sources, but the following were most influential:
- [Khronos Vulkan® Tutorial](https://docs.vulkan.org/tutorial/latest)
- [Vulkan YouTube Channel](https://www.youtube.com/@Vulkan/videos), specifically the `Vulkanised` series has good presentations
- [Sascha Willems Samples](https://github.com/SaschaWillems/Vulkan/)
- [Vulkan Synchronization Examples](https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples)
- *The Modern Vulkan Cookbook* (2024) by P. Kakkar and M. Maurer
  * Referenced source code is available at <https://github.com/PacktPublishing/The-Modern-Vulkan-Cookbook>.

### License
This project is licensed under the MIT License.