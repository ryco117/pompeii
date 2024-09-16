#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

// This shader is invoked when the ray does not intersect any surface.
// It simply returns the background color based on the direction of the ray.
void main() {
  // TODO: Use a texture to allow the caller to specify the background color.
  const vec3 direct_lighting_direction = normalize(vec3(-1, 1, -1));
  const vec3 direct_lighting_color = vec3(1.0);

  const float light_product = dot(direct_lighting_direction, ray_payload.direction);
  const float AMBIENT_SKY = 0.4;
  const vec3 ground_sky_color = abs(ray_payload.direction) * (AMBIENT_SKY + (1.0 - AMBIENT_SKY) * (0.5 * light_product + 0.5));
  const vec3 color_mix = mix(ground_sky_color, direct_lighting_color, max(64.0*light_product - 63.0, 0.0));
  ray_payload.radiance.rgb += color_mix * ray_payload.throughput;
  ray_payload.exit = true;
}
