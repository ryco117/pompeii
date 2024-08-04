#version 460
#extension GL_EXT_ray_tracing : require

// The location index of the shared data here must be consistent across shaders.
layout(location = 1) rayPayloadInEXT bool in_shadow;

void main() {
  in_shadow = false;
}
