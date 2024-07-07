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

layout(push_constant) uniform PushConstants {
  // GPU buffer references.
  VelocityTexture velocity;
  DyeTexture dye;

  ivec2 screen_size;
  uint display_texture;
} push_constants;

layout(location = 0) out vec4 out_color;

void main() {
  const ivec2 coord = ivec2(gl_FragCoord.xy);
  const int index = coord.y * push_constants.screen_size.x + coord.x;

  // For a standard visualization, simply output the dye texture.
  if (push_constants.display_texture == 0) {
    out_color = pow(push_constants.dye.colors[index], vec4(0.575));
  } else {
    out_color = vec4(abs(push_constants.velocity.v[index]) / 1500.0, 0.0, 1.0);
  }
}
