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
  float alpha_cutoff;
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

  // Interpolate the intersection point values according to the barycentric coordinate.
  vec3 position = triangle[0].position_n.xyz*barycentric_coords.x + triangle[1].position_n.xyz*barycentric_coords.y + triangle[2].position_n.xyz*barycentric_coords.z;
  vec2 uv = triangle[0].uv*barycentric_coords.x + triangle[1].uv*barycentric_coords.y + triangle[2].uv*barycentric_coords.z;
  vec3 normal = normalize(
    vec3(triangle[0].position_n.w, triangle[0].ormal)*barycentric_coords.x +
    vec3(triangle[1].position_n.w, triangle[1].ormal)*barycentric_coords.y +
    vec3(triangle[2].position_n.w, triangle[2].ormal)*barycentric_coords.z);
  vec4 tangent = triangle[0].tangent*barycentric_coords.x + triangle[1].tangent*barycentric_coords.y + triangle[2].tangent*barycentric_coords.z;

  // Convert relevant vectors to world-space.
  const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1));
  const vec3 world_normal = normalize(vec3(gl_ObjectToWorldEXT * vec4(normal, 0)));
  const vec4 world_tangent = vec4(vec3(gl_ObjectToWorldEXT * vec4(tangent.xyz, 0)), tangent.w);

  // Calculate the base color of the intersection point.
  vec4 color = vec4(geometry_node.base_color.rgb, 1);
  if (geometry_node.texture_color != MISSING_INDEX) {
    vec4 c = texture(nonuniformEXT(sampler2D(
          textures[geometry_node.texture_color],
          samplers[geometry_node.color_sampler])), uv);
    color = vec4(color.rgb * srgb_to_linear(c.rgb), c.a);
  }

  // Determine the normal associated the the intersection point (given the vertex data).
  const vec3 point_tangent = normalize(world_tangent.xyz);
  const vec3 bitangent = world_tangent.w * cross(world_normal, point_tangent);
  const mat3 tangent_space = mat3(point_tangent, bitangent, world_normal);

  // Update normal based on an optional normal map.
  vec3 point_normal;
  if (geometry_node.texture_normal == MISSING_INDEX) {
    point_normal = world_normal;
  } else {
    vec4 normal_map = texture(nonuniformEXT(sampler2D(
          textures[geometry_node.texture_normal],
          samplers[geometry_node.color_sampler])), uv);
    vec3 tangent_normal = normal_map.xyz * 2.0 - 1.0;
    point_normal = normalize(tangent_space * tangent_normal);
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
  const float roughness_sqr = roughness * roughness;

  // TODO: Understand what this is for and how it should be populated.
  const float material_ior = 1.5;

  // Allow for material refraction.
  float eta = material_ior;

  // Determine the effect of the material's specular component.
  float dielectric_specular = (material_ior - 1.0) / (material_ior + 1.0);
  dielectric_specular *= dielectric_specular;
  const vec3 specular_color = mix(vec3(dielectric_specular), color.rgb, metallic);

  vec3 direct_lighting_color = vec3(1.0);

  // TODO: Use environment mapping when available
  vec3 light_direction = normalize(vec3(-1, 1, -1));
  float light_intensity = 1.0;

  // Trace a ray to determine if this point is in the direct light's shadow.
  in_shadow = true; // Set to true because the miss shader wil set it to `false` if the ray hits the direct light source.
  uint trace_flags = gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT;
  const uint miss_shader_index = 1;
  const int payload_index = 1;
  traceRayEXT(
    top_level_accel_struct,
    trace_flags,
    0xFF, 0, 0,
    miss_shader_index,
    world_position,
    RAY_MIN_DISTANCE,
    light_direction,
    RAY_MAX_DISTANCE,
    payload_index);

  // Apply basic lighting to the intersection point.
  const float AMBIENT_INTENSITY = 0.3;
  if (in_shadow) {
    color.rgb *= AMBIENT_INTENSITY;
  } else {
    color.rgb *= AMBIENT_INTENSITY + (1.0 - AMBIENT_INTENSITY) * max(dot(world_normal, light_direction), 0.0);
  }

  // Sample the hemisphere tangent to this point to apply ambient occlusion.
  float ambient_occlusion = 1.0;
  if (ray_payload.bounce_count == 0) {
    const uint AMBIENT_OCCLUSION_SAMPLES = 12;

    // A uniform distribution of points using an icosahedron.
    vec3 hemisphere_samples[AMBIENT_OCCLUSION_SAMPLES] = vec3[](
      vec3(0.0, 0.0, 1.0),
      vec3(0.7236, 0.5257, 0.4472),
      vec3(-0.2764, 0.8506, 0.4472),
      vec3(-0.8944, 0.0, 0.4472),
      vec3(-0.2764, -0.8506, 0.4472),
      vec3(0.7236, -0.5257, 0.4472),
      vec3(0.2764, 0.8506, -0.4472),
      vec3(-0.7236, 0.5257, -0.4472),
      vec3(-0.7236, -0.5257, -0.4472),
      vec3(0.2764, -0.8506, -0.4472),
      vec3(0.8944, 0.0, -0.4472),
      vec3(0.0, 0.0, -1.0));

    const float INV_OCCLUSION_SAMPLES = 2.0 / float(AMBIENT_OCCLUSION_SAMPLES);
    for (uint i = 0; i < AMBIENT_OCCLUSION_SAMPLES; ++i) {
      const vec3 sample_direction = hemisphere_samples[i];
      if (dot(sample_direction, world_normal) <= 0.0) {
	continue;
      }

      // Trace a ray to determine if this direction is occluded.
      in_shadow = true;
      traceRayEXT(
        top_level_accel_struct,
        trace_flags,
        0xFF, 0, 0,
        miss_shader_index,
        world_position,
        RAY_MIN_DISTANCE,
        sample_direction,
        0.5, // TODO: Determine a reasonable max distance for ambient occlusion.
        payload_index);

      if (in_shadow) {
        ambient_occlusion -= INV_OCCLUSION_SAMPLES;
      }
    }
  }

  // TODO: Determine proper usage of metallic and roughness values.
  vec3 throughput = ray_payload.throughput;
  if (metallic > 0.75 || roughness_sqr < 0.2) {
    // Reflect the ray off the surface.
    ray_payload.origin = world_position;
    ray_payload.direction = reflect(ray_payload.direction, point_normal);
    throughput = ray_payload.throughput * clamp(sqrt(roughness) * (1.0 - metallic*0.25), 0, 1);
  }

  // Apply this triangle's color to the ray payload.
  ray_payload.radiance.rgb += color.rgb * throughput * (0.2 + 0.8 * ambient_occlusion);

  ray_payload.throughput -= throughput;
  if (dot(ray_payload.throughput, ray_payload.throughput) < 0.1) {
    ray_payload.exit = true;
  }
}
