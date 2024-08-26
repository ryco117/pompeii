// Implementatin of a basic incompressible, homogeneous fluid simulation. See https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf, https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu, https://github.com/PavelDoGreat/WebGL-Fluid-Simulation/tree/master.
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 0, rg32f) readonly uniform image2D velocity;
layout(set = 0, binding = 1, rgba32f) readonly uniform image2D dye;
layout(set = 0, binding = 2, r32f) readonly uniform image2D pressure_texture;

layout(scalar, push_constant) uniform PushConstants {
  ivec2 screen_size;
  uint display_texture;
} push_constants;

layout(location = 0) out vec4 out_color;

vec3 color_wheel(float t) {
  return vec3(
    max(sin(t - 0.625) + 0.5, 0) * (2.0 / 3.0),
    max(sin(-t - 0.425) + 0.2, 0) * (5.0 / 6.0),
    max(cos(t - 0.1) - 0.3, 0)) * (1.0 / 0.7);
}

void main() {
  // Oddly, some platforms may give a gl_FragCoord that is out of screen bounds.
  const ivec2 pixel_coord = ivec2(gl_FragCoord.xy);
  if(pixel_coord.x >= push_constants.screen_size.x || pixel_coord.y >= push_constants.screen_size.y) {
    out_color = vec4(1, 0.2, 1, 1);
    return;
  }

  if(push_constants.display_texture == 0) {
    // Dye color.
    out_color = pow(imageLoad(dye, pixel_coord), vec4(vec3(0.55), 1));
  } else if(push_constants.display_texture == 1) {
    // Velocity magnitudes.
    out_color = vec4(abs(imageLoad(velocity, pixel_coord).xy) / 1200.0, 0, 1);
  } else if(push_constants.display_texture == 2) {
    // Pressure.
    float p = clamp(imageLoad(pressure_texture, pixel_coord).x / 220.0, -1, 1);
    out_color = vec4(vec3(pow(max(p, 0), 0.75)) + vec3(0, 0, pow(-min(p, 0), 0.75)), 1);
  } else {
    // Velocity direction.
    const float SIZE = 32;
    const vec2 SHIFT = vec2(0.5);
    const vec2 pixel_velocity = imageLoad(velocity, pixel_coord).xy;
    const vec2 ij = gl_FragCoord.xy / SIZE - SHIFT;
    const vec2 ij_center = round(ij);

    // Normalize the velocity at the center of the grid-cell this pixel belongs to.
    vec2 v = imageLoad(velocity, ivec2((ij_center + SHIFT) * SIZE)).xy;
    v = v / (length(v) + 0.0001);

    // Velocity cone.
    if(push_constants.display_texture == 3) {
      const vec2 diff = 2.0 * (ij - ij_center);
      const float d = length(dot(vec2(v.y, -v.x), diff));
      const float s = clamp(1.875 * dot(v, diff) - d, 0, 1) * exp(-9.0 * max(length(diff) - 0.9, 0));
      const vec3 color = color_wheel(atan(pixel_velocity.y, pixel_velocity.x));
      out_color = vec4(mix(vec3(0, 0, 0.1), color, s), 1);
    }

    // Arrow diffusion of the velocity direction.
    if(push_constants.display_texture == 4) {
      const float d = length(dot(vec2(v.y, -v.x), ij - ij_center));
      const float s = exp(-28.0 * max(d - 0.01, 0));
      const vec3 color = 0.425 * color_wheel(atan(pixel_velocity.y, pixel_velocity.x));
      out_color = vec4(mix(color, vec3(0.85), s), 1);
    }

    // Round diffusion of the velocity direction.
    if(push_constants.display_texture == 5) {
      const vec2 d = ij - (ij_center + 0.35 * v);
      const float s = exp(-6.0 * max(length(d) - 0.1, 0));
      out_color = vec4(s * color_wheel(atan(pixel_velocity.y, pixel_velocity.x)), 1);
    }

    // Gradient-style velocity direction.
    if(push_constants.display_texture == 6) {
      const vec2 d = (ij - ij_center) * 1.41421356;
      const float s = pow(dot(d, v), 0.675);
      out_color = vec4(s * color_wheel(atan(pixel_velocity.y, pixel_velocity.x)), 1);
    }
  }
}
