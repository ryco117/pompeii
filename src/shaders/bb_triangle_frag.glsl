#version 460

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
  float time;
} pushConstants;

void main() {
  outColor = fract(inColor + 0.5*pushConstants.time*vec4(1.0, 0.6, 0.3, 0.0));
}