// Implementatin of a basic incompressible, homogeneous fluid simulation. See https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf, https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu, https://github.com/PavelDoGreat/WebGL-Fluid-Simulation/tree/master.
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(scalar, buffer_reference, buffer_reference_align = 16) readonly buffer VelocityTexture {
  vec2 v[];
};
layout(scalar, buffer_reference, buffer_reference_align = 16) readonly buffer DyeTexture {
  vec4 colors[];
};
layout(scalar, buffer_reference, buffer_reference_align = 16) readonly buffer PressureTexture {
  float pressure[];
};

layout(scalar, push_constant) uniform PushConstants {
  // GPU buffer references.
  VelocityTexture velocity;
  DyeTexture dye;
  PressureTexture pressure;

  ivec2 screen_size;
  uint display_texture;
} push_constants;

layout(location = 0) out vec4 out_color;

int texture_index(ivec2 uv) {
  uv = clamp(uv, ivec2(0), push_constants.screen_size - ivec2(1));
  return uv.y * push_constants.screen_size.x + uv.x;
}

vec3 color_wheel(float t) {
  return vec3(
    max(sin(t - 0.7) + 0.5, 0.0) * (2.0 / 3.0),
    max(sin(-t - 0.3) + 0.2, 0.0) * (5.0 / 6.0),
    max(cos(t - 0.1), 0.0));
}

void main() {
  if(push_constants.display_texture == 0) {
    // Dye color.
    const int index = texture_index(ivec2(gl_FragCoord.xy));
    out_color = pow(push_constants.dye.colors[index], vec4(0.575));
  } else if(push_constants.display_texture == 1) {
    // Velocity magnitudes.
    const int index = texture_index(ivec2(gl_FragCoord.xy));
    out_color = vec4(abs(push_constants.velocity.v[index]) / 1200.0, 0, 1);
  } else if(push_constants.display_texture == 2) {
    // Pressure.
    const int index = texture_index(ivec2(gl_FragCoord.xy));
    float p = clamp(push_constants.pressure.pressure[index] / 220.0, -1, 1);
    out_color = vec4(vec3(pow(max(p, 0), 0.75)) + vec3(0, 0, pow(-min(p, 0), 0.75)), 1);
  } else {
    // Velocity direction.
    const float SIZE = 16.0;
    const vec2 HALF = vec2(0.5);
    const vec2 pixel_velocity = push_constants.velocity.v[texture_index(ivec2(gl_FragCoord.xy))];
    const vec2 ij = gl_FragCoord.xy / SIZE - HALF;
    const vec2 ij_center = round(ij);

    const int index = texture_index(ivec2((ij_center + HALF) * SIZE));
    vec2 v = push_constants.velocity.v[index];
    v = v / (length(v) + 0.0001);
    const vec2 d = ij - (ij_center + 0.35 * v);
    const float s = exp(-2.64 * length(d));
    // const vec2 d = (ij - ij_center) * 1.414;
    // const float s = dot(d, v);
    // out_color = vec4(s * color_wheel(atan(v.y, v.x)), 1.0);
    out_color = vec4(s * color_wheel(atan(pixel_velocity.y, pixel_velocity.x)), 1.0);
  }
}
