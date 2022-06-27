#pragma once

namespace plnoct {

namespace fixed_render_options {
namespace {

// PlenOctree rendering options in [Yu et al. ICCV 2021]
// the values are fixed in this project
constexpr float step_size = 1e-4f;
constexpr float sigma_thresh = 1e-2f;
constexpr float stop_thresh = 1e-2f;
constexpr float background_brightness = 1.0f;

} // namespace
} // namespace fixed_render_options

struct RenderOptions {
    // posterization
    bool posterize = false;
    int num_bilateral_iter = 4;
    int alpha_thresh = 80;
    float step_delta = 0.2f;

    // bilateral filtering
    int filter_radius = 2;
    float sigma_spatial = 3.0f;
    float sigma_CIELAB = 4.25f;
};

} // namespace plnoct