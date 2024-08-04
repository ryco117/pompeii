#ifndef RAY_SHARED_GLSL
#define RAY_SHARED_GLSL 1

// Define the ray payload structure used to track the state of an individual ray.
struct RayPayload {
  vec3 radiance;
  vec3 throughput;
  uint bounce_count;
  uint seed;
  bool exit;
};

// Helper to convert a single channel from a linear color space to sRGB.
float linear_to_srgb(float x) {
  return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 0.41666) - 0.055;
}

// Helper to convert an RGB color from a linear color space to sRGB.
vec3 linear_to_srgb(vec3 rgb) {
  return vec3(linear_to_srgb(rgb.x), linear_to_srgb(rgb.y), linear_to_srgb(rgb.z));
}

#endif
