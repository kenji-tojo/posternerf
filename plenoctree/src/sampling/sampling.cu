#include "sampling.h"

#include <iostream>
#include <stdexcept>

#include "math_utils.h"
#include "render_options.h"


using plnoct::TreeSpec;
using plnoct::Vector3;
using plnoct::Ray;

namespace {

template <typename VEC3>
__device__ inline bool outside_aabb_unit(const VEC3 &__restrict__ pos) {
    return (pos[0] < 0.0 || pos[0] >= 1.0) ||
           (pos[1] < 0.0 || pos[1] >= 1.0) ||
           (pos[2] < 0.0 || pos[2] >= 1.0);
}

__constant__ float g_dirs[128 * 3];
__constant__ float g_scaling[3];

template <typename REAL>
__device__ inline void sample_direction_single(
    Ray<Vector3<REAL>> ray,
    const float *__restrict__ basis_fn,
    const TreeSpec &__restrict__ tree,
    REAL *__restrict__ rgb_out,
    REAL &__restrict__ weight_out)
{
    using Vec3 = Vector3<REAL>;
    namespace opts = plnoct::fixed_render_options;

    weight_out = -1.0;
    if (::outside_aabb_unit(ray.org))
        return;

#pragma unroll
    for (int i=0; i<3; i++)
        ray.dir[i] *= ::g_scaling[i];

    const REAL delta_scale = 1.0f / ray.dir.norm();
    ray.dir *= delta_scale;

    const Vec3 invdir{ 1.0f / (ray.dir[0] + 1e-9f),
                       1.0f / (ray.dir[1] + 1e-9f),
                       1.0f / (ray.dir[2] + 1e-9f) };

    REAL tmin, tmax;
    plnoct::itc_ray_aabb_world(ray.org, invdir, tmin, tmax);
    if (tmax < 0 || tmin != 0)
        return;

    Vec3 pos;
    int64_t node_index;
    REAL sigma_org = 1.0;
    REAL transmittance = 1.0;
    REAL t = tmin;
    REAL cube_sz;

    while (t < tmax) {
#pragma unroll
        for (int i=0; i<3; ++i)
            pos[i] = ray.org[i] + t * ray.dir[i];

        plnoct::octree_query(tree.child, pos, node_index, cube_sz);
        const auto tree_val = tree.data + tree.data_dim * node_index;

        const REAL t_subcube = plnoct::itc_ray_aabb_unit<REAL>(pos, invdir) / cube_sz;
        const REAL delta_t = t_subcube + opts::step_size;
        const float sigma = __half2float(tree_val[tree.data_dim-1]);

        if (t == 0) {
            sigma_org = sigma;
#pragma unroll
            for (int ic=0; ic<3; ++ic) {
                const int off = tree.basis_dim * ic;

                REAL tmp = 0;
                for (int ib=0; ib<tree.basis_dim; ++ib)
                    tmp += basis_fn[ib] * __half2float(tree_val[off+ib]);

                rgb_out[ic] = 1.0f / (1.0f + expf(-tmp));
            }
        }

        if (sigma > opts::sigma_thresh) {
            const REAL att = expf(-delta_t * delta_scale * sigma);
            transmittance *= att;

            if (transmittance < opts::stop_thresh) {
                weight_out = 0;
                return;
            }
        }

        t += delta_t;
    }

    weight_out = sigma_org * transmittance;
}

template<typename REAL>
__global__ void sample_radiance_kernel(
        const size_t num_points,
        const float *__restrict__ points,
        const size_t num_dirs_final,
        const size_t num_sample_per_dirs,
        TreeSpec tree,
        REAL *__restrict__ colors_raw,
        REAL *__restrict__ colors_aa,
        REAL *__restrict__ weights)
{
    extern __shared__ float basis_fn_shared[];
    using Vec3 = Vector3<REAL>;

    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_points)
        return;

    const auto num_dirs_total = num_dirs_final * num_sample_per_dirs;

    Vec3 dir;
    if (threadIdx.x < num_dirs_final * num_sample_per_dirs) {
        plnoct::copy3(dir, ::g_dirs + threadIdx.x * 3);
        dir *= -1.0; // view direction
        eval_sh_basis(dir, tree.basis_dim, basis_fn_shared + threadIdx.x * tree.basis_dim);
    }
    __syncthreads();

    const Vec3 org{ points + tid * 3 };
    REAL w, w_tmp, c[3], c_tmp[3];

    for (int i = 0; i < num_dirs_final; ++i) {
        c[0] = c[1] = c[2] = 0.0;
        w = 0.0;

        for (int j = 0; j < num_sample_per_dirs; ++j) {
            const auto k = i * num_sample_per_dirs + j;
            plnoct::copy3(dir, ::g_dirs + k * 3);
            ::sample_direction_single({ org, dir },
                                      basis_fn_shared + tree.basis_dim * k,
                                      tree,
                                      c_tmp, w_tmp);
            c[0] += w_tmp * c_tmp[0];
            c[1] += w_tmp * c_tmp[1];
            c[2] += w_tmp * c_tmp[2];
            w += w_tmp;

            const auto dst = tid * num_dirs_total + i * num_sample_per_dirs + j;
            colors_raw[dst * 3 + 0] = c_tmp[0];
            colors_raw[dst * 3 + 1] = c_tmp[1];
            colors_raw[dst * 3 + 2] = c_tmp[2];
        }

        REAL denom = w > 0.0 ? w : 1.0;
        const auto dst = tid * num_dirs_final + i;
        colors_aa[dst * 3 + 0] = c[0] / denom;
        colors_aa[dst * 3 + 1] = c[1] / denom;
        colors_aa[dst * 3 + 2] = c[2] / denom;
        weights[dst] = w / (REAL)4.0;
    }
}


// a tiny device memory manager
template <typename T>
struct dev_ptr {
public:
    dev_ptr(size_t count, const T *src) : m_size(count) {
        cudaMalloc((void **)&m_data, count * sizeof(T));
        cudaMemcpy(m_data, src, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    explicit dev_ptr(size_t count) : m_size(count) {
        cudaMalloc((void **)&m_data, count * sizeof(T));
    }

    ~dev_ptr() { if (m_data != nullptr) cudaFree(m_data); }

    void to_host(T *dst) const {
        cudaMemcpy(dst, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    [[nodiscard]] inline T *data() { return m_data; }
    [[nodiscard]] inline T *size() { return m_size; }
private:
    T *m_data = nullptr;
    size_t m_size;
};

} // namespace

namespace plnoct {

void sample_radiance_kernel(const std::vector<float> &points, // spatial sampling points
                            const size_t num_dirs_final,      // number of final anti-aliased sampling directions
                            const size_t num_samples_per_dir, // number of samples per each direction
                            const float *dirs,                // contains (num_dirs_final * num_samples_per_dir) directions
                            TreeSpec &&tree,                  // plenoctree modes
                            const float scale[3],             // scaling of the plenoctree model
                            float *colors_raw,                // raw radiance samples in RGB
                            float *colors_aa,                 // anti-aliased radiance samples in RGB
                            float *weights)                   // visibility weights for each anti-aliased radiance sample
{
    if (num_dirs_final * num_samples_per_dir > 128) {
        std::cerr << "too large total number of directions: " << num_dirs_final * num_samples_per_dir << std::endl;
        throw std::runtime_error("number of directions should be <= 128");
    }

    const auto num_points = points.size() / 3;
    const auto num_dirs_total = num_dirs_final * num_samples_per_dir;

    ::dev_ptr<float> points_CUDA{ points.size(), points.data() };
    ::dev_ptr<float> colors_raw_CUDA{ num_points * num_dirs_total * 3 };
    ::dev_ptr<float> colors_aa_CUDA{ num_points * num_dirs_final * 3 };
    ::dev_ptr<float> weights_CUDA{ num_points * num_dirs_final };

    cudaMemcpyToSymbol(::g_dirs, dirs, num_dirs_total * 3 * sizeof(float));
    cudaMemcpyToSymbol(::g_scaling, scale, 3 * sizeof(float));

    const int num_threads_per_block = 512;
    const int num_blocks = 1 + int(num_points-1)/num_threads_per_block;
    const size_t shared_size = num_dirs_total * tree.basis_dim * sizeof(float);

    ::sample_radiance_kernel<<<num_blocks, num_threads_per_block, shared_size>>>(
            num_points,
            points_CUDA.data(),
            num_dirs_final,
            num_samples_per_dir,
            tree,
            colors_raw_CUDA.data(), colors_aa_CUDA.data(), weights_CUDA.data());

    colors_raw_CUDA.to_host(colors_raw);
    colors_aa_CUDA.to_host(colors_aa);
    weights_CUDA.to_host(weights);
}

} // namespace plnoct