#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

// This shader is invoked when the ray does not intersect any surface.
// It simply returns the background color based on the direction of the ray.
void main() {
  // TODO: Use a texture to allow the caller to specify the background color.
  ray_payload.radiance.rgb += abs(ray_payload.direction) * ray_payload.throughput;
  ray_payload.exit = true;
}
