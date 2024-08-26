#ifndef FLUID_SHARED_GLSL
#define FLUID_SHARED_GLSL 1

// The push constants are passed to each compute shader used in the fluid simulation demo.
layout(scalar, push_constant) uniform PushConstants {
  vec4 cursor_dye;
  vec2 cursor_position;
  vec2 cursor_velocity;
  ivec2 screen_size;
  float delta_time;
  float velocity_diffusion_rate;
  float dye_diffusion_rate;
  float vorticity_strength; // Sane values are 0 to 50. Default is 30.
} push_constants;

#endif
