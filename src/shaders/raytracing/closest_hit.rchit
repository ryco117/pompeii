#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;
layout(location = 1) rayPayloadEXT bool in_shadow;

// Geometry-dependent intersection attributes.
hitAttributeEXT vec2 hit_attributes;

// layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_accel_struct;

// layout(scalar, push_constant) uniform PushConstants {
//   mat4 view_inverse;
//   float time;
// } push_constants;

void main() {
  const vec3 barycentric_coords = vec3(1.0 - hit_attributes.x - hit_attributes.y, hit_attributes.x, hit_attributes.y);

  // TODO: Implement shadow rays.
  ray_payload.radiance = min(vec3(1.0), abs(barycentric_coords));
}
