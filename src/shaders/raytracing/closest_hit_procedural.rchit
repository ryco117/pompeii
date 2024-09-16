#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;
layout(location = 1) rayPayloadEXT bool in_shadow;

// The world space position of the hit point.
hitAttributeEXT vec2 hit_attribute;

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_accel_struct;

layout(buffer_reference, buffer_reference_align = 16, std430) readonly buffer Sphere {
  vec4 center_radius;
  vec4 color;
};
layout(set = 3, binding = 0, std430) readonly buffer SphereBuffers { Sphere s[]; } sphere_buffers;

void main() {
  Sphere sphere = sphere_buffers.s[gl_InstanceCustomIndexEXT];
  const vec3 light_direction = normalize(vec3(-1, 1, -1));
  const vec3 world_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  // Convert the hit attribute to a `vec3` normal from spherical coordinates.
  const float sin_phi = sin(hit_attribute.y);
  const vec3 hit_normal = vec3(cos(hit_attribute.x) * sin_phi, sin(hit_attribute.x) * sin_phi, cos(hit_attribute.y));

  const vec3 new_ray_direction = reflect(gl_WorldRayDirectionEXT, hit_normal);
  const vec3 new_position = world_position + new_ray_direction * RAY_MIN_DISTANCE;

  // Trace a ray to determine if this point is in the direct light's shadow.
  in_shadow = true; // Set to true because the miss shader wil set it to `false` if the ray hits the direct light source.
  float light_dot = dot(hit_normal, light_direction);

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
      new_position,
      RAY_MIN_DISTANCE,
      light_direction,
      RAY_MAX_DISTANCE,
      payload_index);
  } else {
    light_dot = 0.0;
  }

  // Apply simple direct shadows and reflection.
  const float AMBIENT_INTENSITY = 0.2;
  const float REFLECTANCE = float(gl_HitKindEXT) / 255.0;
  // const float REFLECTANCE = 0.0;
  float light_factor;
  if (in_shadow) {
    light_factor = AMBIENT_INTENSITY;
  } else {
    light_factor = AMBIENT_INTENSITY + (1.0 - AMBIENT_INTENSITY) * light_dot;
  }

  uint seed = gl_HitKindEXT;
  const vec3 hit_color = gl_HitKindEXT == 0 ? sphere.color.rgb : randomHemispherePoint(seed, normalize(vec3(1)));
  // const vec3 hit_color = (vec3(5.6) + new_position)/11.2;
  ray_payload.radiance += (1.0 - REFLECTANCE) * hit_color * light_factor * ray_payload.throughput;

  // Setup for reflection ray.
  ray_payload.origin = new_position;
  ray_payload.direction = new_ray_direction;

  ray_payload.throughput = REFLECTANCE * ray_payload.throughput;
}
