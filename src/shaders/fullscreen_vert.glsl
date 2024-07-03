#version 460
vec2 positions[4] = { vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0) };

vec2 tex_coords[4] = { vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 0.0) };

layout(location = 0) out vec2 out_tex_coord;

void main() {
  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
  out_tex_coord = tex_coords[gl_VertexIndex];
}