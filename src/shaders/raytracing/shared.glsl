#ifndef RAY_SHARED_GLSL
#define RAY_SHARED_GLSL 1

// Define the ray payload structure used to track the state of an individual ray.
struct RayPayload {
  // The color and ambient occlusion accumulated by the ray.
  vec4 radiance;

  // The amount of transmission energy left in the ray.
  vec3 throughput;

  // Store the direction and origin to allow for bounce calculations.
  vec3 origin;
  vec3 direction;

  // The current ray depth.
  uint bounce_count;

  // The current random seed.
  uint seed;

  // Whether to exit the ray tracing loop.
  bool exit;
};

// The shared ray distance constants.
const float RAY_MIN_DISTANCE = 0.001;
const float RAY_MAX_DISTANCE = 10000.0;

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

// this is a hash function for generating pseudo-random numbers. Taken from
// here: http://jcgt.org/published/0009/03/02/
uvec3 pcg_uvec3_uvec3(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> 16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

// Lightweight Java Game Library License
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// * Neither the name Lightweight Java Game Library nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// From:
// https://github.com/LWJGL/lwjgl3-demos/blob/main/res/org/lwjgl/demo/opengl/raytracing/randomCommon.glsl
vec3 randomSpherePoint(vec3 rand) {
  const float PI = 3.141592653589793;
  float ang1 = (rand.x + 1.0) * PI;  // [-1..1) -> [0..2*PI)
  float u = rand.y;  // [-1..1), cos and acos(2v-1) cancel each other out, so we
                     // arrive at
                     // [-1..1)
  float u2 = u * u;
  float sqrt1MinusU2 = sqrt(1.0 - u2);
  float x = sqrt1MinusU2 * cos(ang1);
  float y = sqrt1MinusU2 * sin(ang1);
  float z = u;
  return vec3(x, y, z);
}
// From:
// https://github.com/LWJGL/lwjgl3-demos/blob/main/res/org/lwjgl/demo/opengl/raytracing/randomCommon.glsl
vec3 randomHemispherePoint(inout uint seed, vec3 n) {
  /**
   * Generate random sphere point and swap vector along the normal, if it
   * points to the wrong of the two hemispheres.
   * This method provides a uniform distribution over the hemisphere,
   * provided that the sphere distribution is also uniform.
   */
  vec3 v = randomSpherePoint(normalize(pcg_uvec3_uvec3(uvec3(seed))));
  seed = seed * 747796405u + 2891336453u;
  return v * sign(dot(v, n));
}

#endif
