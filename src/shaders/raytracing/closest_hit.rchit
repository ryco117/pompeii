#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "shared.glsl"
#include "random.glsl"
#include "pbr_gltf.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;
layout(location = 1) rayPayloadEXT bool in_shadow;

// Geometry-dependent intersection attributes.
hitAttributeEXT vec2 hit_attributes;

// The format of vertex data used in this pipeline.
struct Vertex {
  vec4 position_n;
  vec2 ormal;
  vec2 uv;
  vec4 tangent;
};
layout(buffer_reference, buffer_reference_align = 16, std430) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, buffer_reference_align = 4, std430) readonly buffer Indices { uint i[]; };

// A single glTF node with its geometry and textures.
struct GeometryNode {
  vec4 base_color;
  uint index_offset;
  uint texture_color;
  uint color_sampler;

  float metallic_factor;
  float roughness_factor;

  uint texture_normal;
  uint texture_metallic;
  uint _padding;
};

// An instance of a glTF model including its vertices and geometry nodes.
layout(buffer_reference, buffer_reference_align = 16, std430) readonly buffer GltfInstance {
  Vertices vertex_buffer;
  Indices index_buffer;
  GeometryNode nodes[];
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_accel_struct;

layout(set = 3, binding = 0, std430) readonly buffer InstanceBuffers { GltfInstance g[]; } instance_buffers;
layout(set = 3, binding = 1) uniform texture2D textures[];
layout(set = 3, binding = 2) uniform sampler samplers[];

const uint MISSING_INDEX = 0xFFFFFFFFu;

void main() {
  const uint triangle_index_offset = 3 * gl_PrimitiveID;
  const vec3 barycentric_coords = vec3(1.0 - hit_attributes.x - hit_attributes.y, hit_attributes.x, hit_attributes.y);

  // Use the instance ID to determine which glTF model instance was hit.
  GltfInstance gltf_instance = instance_buffers.g[gl_InstanceID];

  // Each material in the instance is placed into a different geometry index in the bottom-level acceleration structure.
  GeometryNode geometry_node = gltf_instance.nodes[gl_GeometryIndexEXT];

  // Determine the properties of the vertices of this triangle.
  Vertex triangle[3];
  const uint index_offset = geometry_node.index_offset;
  for (uint i = 0; i < 3; ++i) {
    const uint vertex_index = gltf_instance.index_buffer.i[index_offset + triangle_index_offset + i];
    triangle[i] = gltf_instance.vertex_buffer.v[vertex_index];
  }

  // Interpolate the normal and UV coordinates of the intersection point.
  vec2 uv = triangle[0].uv*barycentric_coords.x + triangle[1].uv*barycentric_coords.y + triangle[2].uv*barycentric_coords.z;
  vec3 normal = normalize(
    vec3(triangle[0].position_n.w, triangle[0].ormal)*barycentric_coords.x +
    vec3(triangle[1].position_n.w, triangle[1].ormal)*barycentric_coords.y +
    vec3(triangle[2].position_n.w, triangle[2].ormal)*barycentric_coords.z);
  vec4 tangent = triangle[0].tangent*barycentric_coords.x + triangle[1].tangent*barycentric_coords.y + triangle[2].tangent*barycentric_coords.z;

  // Determine the normal in world space.
  const vec3 world_normal = normalize(vec3(normal*gl_WorldToObjectEXT));

  // Calculate the base color of the intersection point.
  vec3 color = geometry_node.base_color.rgb;
  if (geometry_node.texture_color != MISSING_INDEX) {
    vec3 c = texture(nonuniformEXT(sampler2D(
          textures[geometry_node.texture_color],
          samplers[geometry_node.color_sampler])), uv).rgb;
    color *= srgb_to_linear(c);
  }

  vec3 point_normal;
  if (geometry_node.texture_normal == MISSING_INDEX) {

  }

  // Determine the metallic-roughness color to use.
  float metallic = geometry_node.metallic_factor;
  float roughness = geometry_node.roughness_factor;
  if (geometry_node.texture_metallic != MISSING_INDEX) {
    vec4 metallic_roughness = texture(nonuniformEXT(sampler2D(
        textures[geometry_node.texture_metallic],
        samplers[geometry_node.color_sampler])), uv);
    metallic *= metallic_roughness.b;
    roughness *= metallic_roughness.g;
  }

  // Avoid artifacts with a sufficiently non-zero roughness.
  roughness = max(roughness, 0.001);

  // TODO: Understand what this is for and how it should be populated.
  const float material_ior = 1.5;

  // Allow for material refraction.
  float eta = material_ior;

  float dielectric_specular = (material_ior - 1.0) / (material_ior + 1.0);
  dielectric_specular *= dielectric_specular;

  vec3 specular_color = mix(vec3(dielectric_specular), color, metallic);
  vec3 direct_lighting_color = vec3(0.0);
  vec3 environment_lighting_color = vec3(0.0);

  // TODO: Use environment mapping when available
  vec3 light_direction = normalize(vec3(1));
  float light_intensity = 1.0;

  // Apply the lighting to the intersection point.
  if (false) {
    float pdf;
    vec3 f = PbrEval(eta, metallic, roughness, color, specular_color,
      -gl_WorldRayDirectionEXT, point_normal, light_direction, pdf);

    float cos_theta = abs(dot(light_direction, point_normal));

    float weight = max(0.0, power_heuristic(light_intensity, pdf));
    if (weight > 0.0) {
      direct_lighting_color += weight * f * cos_theta * environment_lighting_color / (light_intensity + 0.001);
    }
  }

  // Apply this triangle's color to the ray payload.
  ray_payload.radiance += color * ray_payload.throughput;
  ray_payload.radiance = clamp(ray_payload.radiance, vec3(0), vec3(1));

  // TODO: Implement shadow rays.
  in_shadow = false;

  // Determine if more rays should be traced.
  ray_payload.exit = true;
}
