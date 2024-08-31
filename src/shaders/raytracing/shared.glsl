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

// Helpers to convert channels from sRGB to a linear color space.
float srgb_to_linear(float x) {
  return x < 0.04045 ? x * 0.0773993808 : pow((x + 0.055) * 0.9478672986, 2.4);
}
vec3 srgb_to_linear(vec3 rgb) {
  return vec3(srgb_to_linear(rgb.x), srgb_to_linear(rgb.y), srgb_to_linear(rgb.z));
}

// Helpers to convert channels from a linear color space to sRGB.
float linear_to_srgb(float x) {
  return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 0.41666666) - 0.055;
}
vec3 linear_to_srgb(vec3 rgb) {
  return vec3(linear_to_srgb(rgb.x), linear_to_srgb(rgb.y), linear_to_srgb(rgb.z));
}

// Helper to calculate a power heuristic weight for light sampling.
float power_heuristic(float a, float b) {
  float a2 = a * a;
  float b2 = b * b;
  return a2 / (a2 + b2);
}

#endif
