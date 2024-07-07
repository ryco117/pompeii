#version 460

layout(push_constant) uniform PushConstants {
  vec2 inverse_screen_size;
} push_constants;

layout(set = 0, binding = 0) uniform sampler2D input_texture;

layout(location = 0) out vec4 out_color;

// Constants used for the FXAA algorithm.
const float EDGE_THRESHOLD_MIN = 1.0 / 20.0;
const float EDGE_THRESHOLD_MAX = 1.0 / 12.0;
const float PIXEL_BLEND_LIMIT  = 0.8;
const float MIN_PIXEL_ALIASING = 1.0 / 8.0;
const float NUM_LOOP_FOR_EDGE_DETECTION = 2;

// Constants to help indexing neighboring cells by name.
const int CENTER = 0;
const int TOP = 1;
const int BOTTOM = 2;
const int LEFT = 3;
const int RIGHT = 4;
const int TOP_RIGHT = 5;
const int BOTTOM_RIGHT = 6;
const int TOP_LEFT = 7;
const int BOTTOM_LEFT = 8;

// Neighbor offsets aligned to the name constants.
const vec2 BLOCK_OFFSETS[] = {
  vec2(0), vec2(0, -1), vec2(0, 1), vec2(-1, 0), vec2(1, 0),
  vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(-1, 1)
};

// Calculate the luminance of a color using a normalized linear combination of the color channels.
float measure_luminance(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

// Traverses an edge in the image bi-directionally in search of the endpoints.
// Returns 1.0 if anti-aliasing should be applied, 0.0 otherwise.
float find_end_point_position(
  vec2 tex_coord,
  float luminance,
  float high_contrast_pixel,
  float step_length,
  vec2 inverse_screen,
  bool is_horizontal,
  out vec2 out_position
) {
  // Initialize the direction and position of the high-contrast pixel.
  vec2 texture_coord_of_high_contrast_pixel = tex_coord;

  vec2 edge_direction;
  if(is_horizontal) {
    texture_coord_of_high_contrast_pixel.y += step_length;
    edge_direction = vec2(inverse_screen.x, 0);
  } else {
    texture_coord_of_high_contrast_pixel.x += step_length;
    edge_direction = vec2(0, inverse_screen.y);
  }

  // Prepare for search loop.
  float high_contrast_pixel_north;
  float high_contrast_pixel_south;
  float pixel_north;
  float pixel_south;
  bool done_going_north = false;
  bool done_going_south = false;
  vec2 position_contrast_north = texture_coord_of_high_contrast_pixel;
  vec2 position_contrast_south = texture_coord_of_high_contrast_pixel;
  vec2 position_pixel_north = tex_coord;
  vec2 position_pixel_south = tex_coord;

  for(int i = 0; i < NUM_LOOP_FOR_EDGE_DETECTION; ++i) {
    // Process searching north.
    if(!done_going_north) {
      // Update the positions of the "high-contrast" and "middle" reference pixels along the edge.
      position_contrast_north -= edge_direction;
      position_pixel_north -= edge_direction;

      high_contrast_pixel_north = measure_luminance(texture(input_texture, position_contrast_north).rgb);
      pixel_north = measure_luminance(texture(input_texture, position_pixel_north).rgb);
      done_going_north = abs(high_contrast_pixel_north - high_contrast_pixel) > abs(high_contrast_pixel_north - luminance)
        || abs(pixel_north - luminance) > abs(pixel_north - high_contrast_pixel);
    }

    // Process searching south.
    if(!done_going_south) {
      // Update the positions of the "high-contrast" and "middle" reference pixels along the edge.
      position_contrast_south += edge_direction;
      position_pixel_south += edge_direction;

      high_contrast_pixel_south = measure_luminance(texture(input_texture, position_contrast_south).rgb);
      pixel_south = measure_luminance(texture(input_texture, position_pixel_south).rgb);
      done_going_south = abs(high_contrast_pixel_south - high_contrast_pixel) > abs(high_contrast_pixel_south - luminance)
        || abs(pixel_south - luminance) > abs(pixel_south - high_contrast_pixel);
    }

    // Check if the search is done.
    if(done_going_north && done_going_south) {
      break;
    }
  }

  // Get the distance to each endpoint.
  float destination_north;
  float destination_south;
  if(is_horizontal) {
    destination_north = tex_coord.x - position_pixel_north.x;
    destination_south = position_pixel_south.x - tex_coord.x;
  } else {
    destination_north = tex_coord.y - position_pixel_north.y;
    destination_south = position_pixel_south.y - tex_coord.y;
  }

  // Determine which endpoint is closer.
  const bool is_north_closer = destination_north < destination_south;
  float min_distance = min(destination_north, destination_south);
  float closer_luminance = is_north_closer ? pixel_north : pixel_south;

  // If the luminance difference to the closer endpoint is sufficiently large, apply anti-aliasing.
  const bool edge_anti_aliasing = abs(closer_luminance - high_contrast_pixel) < abs(closer_luminance - luminance);

  // Calculate the pixel position to be used for anti-aliasing the original pixel.
  const float negative_inverse_edge_distance = -1.0 / (destination_north + destination_south);
  const float pixel_offset = min_distance * negative_inverse_edge_distance + 0.5;
  out_position = tex_coord;
  if(is_horizontal) {
    out_position.y += pixel_offset * step_length;
  } else {
    out_position.x += pixel_offset * step_length;
  }

  return edge_anti_aliasing ? 1.0 : 0.0;
}

// Apply the FXAA algorithm to the input texture which is usually the result of the render before post-processing.
vec4 apply_fxaa(vec2 screen_coord) {
  const vec2 inverse_screen = push_constants.inverse_screen_size;
  const vec2 tex_coord = screen_coord * inverse_screen;

  float min_luminance = 10000000;
  float max_luminance = 0;
  vec4 block_color_and_luminance[9];
  vec3 color_sum = vec3(0);

  for(int i = 0; i < 9; ++i) {
    block_color_and_luminance[i].xyz = texture(input_texture, tex_coord + BLOCK_OFFSETS[i] * inverse_screen).rgb;
    color_sum += block_color_and_luminance[i].xyz;

    block_color_and_luminance[i].w = measure_luminance(block_color_and_luminance[i].xyz);

    // Ignore the corners from the min and max luminance calculation.
    if(i < 5) {
      min_luminance = min(min_luminance, block_color_and_luminance[i].w);
      max_luminance = max(max_luminance, block_color_and_luminance[i].w);
    }
  }

  // If there is little different in luminance between the neighboring pixels, return the center pixel.
  const float range_luminance = max_luminance - min_luminance;
  if(range_luminance < max(EDGE_THRESHOLD_MIN, EDGE_THRESHOLD_MAX * max_luminance)) {
    return vec4(block_color_and_luminance[CENTER].xyz, 1);
  }

  // Calculate the pixel blend amount.
  const float luminance_top_bottom = block_color_and_luminance[TOP].w + block_color_and_luminance[BOTTOM].w;
  const float luminance_left_right = block_color_and_luminance[LEFT].w + block_color_and_luminance[RIGHT].w;
  const float luminance_top_corners = block_color_and_luminance[TOP_LEFT].w + block_color_and_luminance[TOP_RIGHT].w;
  const float luminance_bottom_corners = block_color_and_luminance[BOTTOM_LEFT].w + block_color_and_luminance[BOTTOM_RIGHT].w;
  const float luminance_left_corners = block_color_and_luminance[TOP_LEFT].w + block_color_and_luminance[BOTTOM_LEFT].w;
  const float luminance_right_corners = block_color_and_luminance[TOP_RIGHT].w + block_color_and_luminance[BOTTOM_RIGHT].w;

  const float tblr = (luminance_top_bottom + luminance_left_right) / 4.0;
  const float average_luminance_diff = abs(block_color_and_luminance[CENTER].w - tblr);
  float pixel_blend = max(0.0, average_luminance_diff / range_luminance - MIN_PIXEL_ALIASING);
  pixel_blend = min(PIXEL_BLEND_LIMIT, pixel_blend * (1.0 / (1.0 - MIN_PIXEL_ALIASING)));

  // Determine the direction of the edge.
  const vec3 average_color = color_sum * (1.0 / 9.0); // Store 1/9 as a constant at compile and perform multiplication at runtime. Same trick is used throughout.
  const float vertical_edge_row_1 = abs(-2.0 * block_color_and_luminance[TOP].w + luminance_top_corners);
  const float vertical_edge_row_2 = abs(-2.0 * block_color_and_luminance[CENTER].w + luminance_left_right);
  const float vertical_edge_row_3 = abs(-2.0 * block_color_and_luminance[BOTTOM].w + luminance_bottom_corners);
  const float vertical_edge = (vertical_edge_row_1 + 2.0 * vertical_edge_row_2 + vertical_edge_row_3) * (1.0 / 12.0);

  const float horizontal_edge_col_1 = abs(-2.0 * block_color_and_luminance[LEFT].w + luminance_left_corners);
  const float horizontal_edge_col_2 = abs(-2.0 * block_color_and_luminance[CENTER].w + luminance_top_bottom);
  const float horizontal_edge_col_3 = abs(-2.0 * block_color_and_luminance[RIGHT].w + luminance_right_corners);
  const float horizontal_edge = (horizontal_edge_col_1 + 2.0 * horizontal_edge_col_2 + horizontal_edge_col_3) * (1.0 / 12.0);

  const bool is_horizontal = horizontal_edge > vertical_edge;

  const float steep_luminance_1 = is_horizontal ? block_color_and_luminance[TOP].w : block_color_and_luminance[LEFT].w;
  const float steep_luminance_2 = is_horizontal ? block_color_and_luminance[BOTTOM].w : block_color_and_luminance[RIGHT].w;

  const bool is_first_steeper = abs(steep_luminance_1 - block_color_and_luminance[CENTER].w) >= abs(steep_luminance_2 - block_color_and_luminance[CENTER].w);
  float step_length = is_horizontal ? -inverse_screen.y : -inverse_screen.x;
  float high_contrast_pixel;
  if(is_first_steeper) {
    high_contrast_pixel = steep_luminance_1;
  } else {
    high_contrast_pixel = steep_luminance_2;
    step_length = -step_length;
  }
  vec3 color_edge_anti_aliasing_pixel = block_color_and_luminance[CENTER].xyz;
  vec2 out_position_for_edge;

  const float res = find_end_point_position(
    tex_coord,
    block_color_and_luminance[CENTER].w,
    high_contrast_pixel,
    step_length,
    inverse_screen,
    is_horizontal,
    out_position_for_edge
  );

  if(res == 1.0) {
    color_edge_anti_aliasing_pixel = texture(input_texture, out_position_for_edge).rgb;
  }

  return vec4(mix(color_edge_anti_aliasing_pixel, average_color, pixel_blend), 1);
}

void main() {
  out_color = apply_fxaa(gl_FragCoord.xy);
}
