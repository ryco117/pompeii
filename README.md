# Pompeii

### Imposed restrictions
- (At least) Vulkan API version 1.3 must be used. This version greatly reduced the extension/feature fragmentation in the ecosystem and makes the engine both concise and portable.
- The chosen physical device must support the following device features. These features are common in modern hardware and allow for better primitives and API usage.
  * `VkPhysicalDeviceBufferDeviceAddressFeatures`
  * `VkPhysicalDeviceDynamicRenderingFeatures`
  * `VkPhysicalDeviceSynchronization2Features`

### License
This project is licensed under the MIT License.