#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;
layout(location = 1) rayPayloadEXT bool in_shadow;

// The world space position of the hit point.
hitAttributeEXT vec3 world_normal;

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_accel_struct;

layout(buffer_reference, buffer_reference_align = 16, std430) readonly buffer Sphere {
  vec4 center_radius;
  vec4 color;
};
layout(set = 3, binding = 0, std430) readonly buffer SphereBuffers { Sphere s[]; } sphere_buffers;

void main() {
  Sphere sphere = sphere_buffers.s[gl_InstanceCustomIndexEXT];
  const vec3 world_position = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
  const vec3 light_direction = normalize(vec3(-1, 1, -1));

  // Trace a ray to determine if this point is in the direct light's shadow.
  in_shadow = true; // Set to true because the miss shader wil set it to `false` if the ray hits the direct light source.
  float light_dot = dot(world_normal, light_direction);

  // If the hit point is facing away from the light, skip the shadow ray.
  if (light_dot > 0.0) {
    uint trace_flags = gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT;
    const uint miss_shader_index = 1;
    const int payload_index = 1;
    traceRayEXT(
      top_level_accel_struct,
      trace_flags,
      0xFF, 0, 0,
      miss_shader_index,
      world_position,
      RAY_MIN_DISTANCE,
      light_direction,
      RAY_MAX_DISTANCE,
      payload_index);
  }

  // Apply simple direct shadows and reflection.
  const float AMBIENT_INTENSITY = 0.2;
  const float REFLECTANCE = 1.0;
  float light_factor;
  if (in_shadow) {
    light_factor = AMBIENT_INTENSITY;
  } else {
    light_dot = max(light_dot, 0.0);
    light_factor = AMBIENT_INTENSITY + (1.0 - AMBIENT_INTENSITY) * light_dot;
  }
  const vec3 light_throughput = light_factor * ray_payload.throughput;
  ray_payload.radiance += (1.0 - REFLECTANCE) * sphere.color.rgb * light_throughput;

  // Setup for reflection ray.
  ray_payload.origin = world_position;
  ray_payload.direction = reflect(ray_payload.direction, world_normal);

  ray_payload.throughput = light_throughput * REFLECTANCE;
}
