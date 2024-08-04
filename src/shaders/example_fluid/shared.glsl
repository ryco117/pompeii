#ifndef FLUID_SHARED_GLSL
#define FLUID_SHARED_GLSL 1

// The push constants are passed to each compute shader used in the fluid simulation demo.
layout(scalar, push_constant) uniform PushConstants {
  // Store GPU pointers to each texture/buffer.
  VelocityTexture input_velocity;
  CurlTexture curl;
  DivergenceTexture divergence;
  PressureTexture alpha_pressure;
  PressureTexture beta_pressure;
  VelocityTexture output_velocity;
  DyeTexture input_dye;
  DyeTexture output_dye;

  vec4 cursor_dye;
  vec2 cursor_position;
  vec2 cursor_velocity;
  ivec2 screen_size;
  float delta_time;
  float velocity_diffusion_rate;
  float dye_diffusion_rate;
  float vorticity_strength; // Sane values are 0 to 50. Default is 30.
} push_constants;

// The shared method for safely indexing into the texture buffers.
int texture_index(ivec2 uv) {
  uv = clamp(uv, ivec2(0), push_constants.screen_size - ivec2(1));
  return uv.y * push_constants.screen_size.x + uv.x;
}

#endif
