#pragma once

#include <vector>
#include <cuda_fp16.h>

#include "data_spec.h"


namespace plnoct {

/*
 * num_points     = points.size() / 3
 * num_dirs_total = num_dirs_final * num_samples_per_dir
 * colors_raw:    float[num_points * num_dirs_total * 3]
 * colors_aa:     float[num_points * num_dirs_final * 3]
 * weights:       float[num_points * num_dirs_final]
 */
void sample_radiance_kernel(const std::vector<float> &points, // spatial sampling points
                            size_t num_dirs_final,            // number of final anti-aliased sampling directions
                            size_t num_samples_per_dir,       // number of samples per each direction
                            const float *dirs,                // contains (num_dirs_final * num_samples_per_dir) directions
                            TreeSpec &&tree,                  // plenoctree modes
                            const float scale[3],             // scaling of the plenoctree model
                            float *colors_raw,                // raw radiance samples in RGB
                            float *colors_aa,                 // anti-aliased radiance samples in RGB
                            float *weights);                  // visibility weights for each anti-aliased radiance sample

} // namespace plnoct