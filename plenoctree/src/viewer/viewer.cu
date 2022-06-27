#include <cstdint>

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <surface_indirect_functions.h>

#include "data_spec.h"
#include "math_utils.h"
#include "render_options.h"
#include "palette.h"


using plnoct::CameraSpec;
using plnoct::TreeSpec;
using plnoct::Vector3;
using plnoct::Ray;

namespace {

// global data
namespace fixed_opts = plnoct::fixed_render_options;
__constant__ plnoct::RenderOptions g_opts;

__constant__ plnoct::Labels g_labels;


// a tiny wrapper of cudaSurfaceObject_t
struct SurfaceObjectImage {
public:
    SurfaceObjectImage(cudaArray_t &CA, int width, int height)
            : m_width(width), m_height(height)
    {
        cudaResourceDesc desc{ };
        ::memset(&desc, 0, sizeof(desc));
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = CA;
        cudaCreateSurfaceObject(&m_surf_obj, &desc);
    }

    __device__ inline void read(int ix, int iy, uint8_t rgba_ub[4]) const {
        surf2Dread(reinterpret_cast<uint32_t *>(rgba_ub),
                   m_surf_obj,
                   ix * 4, iy,
                   cudaBoundaryModeZero);
    }

    __device__ inline void write(int ix, int iy, const uint8_t rgba_ub[4]) const {
        surf2Dwrite(*reinterpret_cast<const uint32_t *>(rgba_ub),
                    m_surf_obj,
                    ix * 4, iy,
                    cudaBoundaryModeZero); // squelches out-of-bound writes
    }

    __device__ inline int width() const { return m_width; }
    __device__ inline int height() const { return m_height; }
private:
    cudaSurfaceObject_t m_surf_obj = 0;
    int m_width;
    int m_height;
};


// raytracing of plenoctree
template<typename REAL>
__device__ inline void raytrace_single(
        Ray<Vector3<REAL>> ray,
        const TreeSpec &__restrict__ tree,
        const Vector3<REAL> &__restrict__ scaling,
        REAL *__restrict__ rgba_out)
{
    using Vec3 = Vector3<REAL>;

    ray.dir *= scaling;
    const REAL delta_scale = 1.0f / ray.dir.norm();
    ray.dir *= delta_scale;

    const Vec3 invdir{ REAL(1.0f / (ray.dir[0] + 1e-9)),
                       REAL(1.0f / (ray.dir[1] + 1e-9)),
                       REAL(1.0f / (ray.dir[2] + 1e-9)) };

    REAL tmin, tmax;
    plnoct::itc_ray_aabb_world(ray.org, invdir, tmin, tmax);
    if (tmax < 0 || tmin > tmax)
        return; // ray doesn't hit the box

    Vec3 pos;
    REAL basis_fn[SH_BASIS_DIM_MAX];
    plnoct::eval_sh_basis(ray.vdir, tree.basis_dim, basis_fn);

    int64_t node_index;
    REAL light_intensity = 1;
    REAL t = tmin;
    REAL cube_sz;

    while (t < tmax) {
#pragma unroll
        for (int i=0; i<3; ++i)
            pos[i] = ray.org[i] + t * ray.dir[i];

        plnoct::octree_query(tree.child, pos, node_index, cube_sz);
        const auto tree_val = tree.data + tree.data_dim * node_index;

        const REAL t_subcube = plnoct::itc_ray_aabb_unit<REAL>(pos, invdir) / cube_sz;
        const REAL delta_t = t_subcube + fixed_opts::step_size;
        const float sigma = __half2float(tree_val[tree.data_dim-1]);

        if (sigma > fixed_opts::sigma_thresh) {
            const REAL att = expf(-delta_t * delta_scale * sigma);
            const REAL weight = light_intensity * (1.0f - att);
#pragma unroll
            for (int ic=0; ic<3; ++ic) {
                const int off = tree.basis_dim * ic;

                REAL tmp = 0;
                for (int ib=0; ib<tree.basis_dim; ++ib)
                    tmp += basis_fn[ib] * __half2float(tree_val[off+ib]);

                rgba_out[ic] += weight / (1.0f + expf(-tmp));
            }
            light_intensity *= att;

            if (light_intensity < fixed_opts::stop_thresh) {
                // full opacity, stop
                const REAL scale = 1.0f / (1.0f - light_intensity);
#pragma unroll
                for (int ic=0; ic<3; ++ic)
                    rgba_out[ic] *= scale;
                rgba_out[3] = 1.0f;
                return;
            }
        }

        t += delta_t;
    }

    rgba_out[3] = 1.0f - light_intensity;
}

template<typename VEC3>
__device__ inline void screen2worlddir(
        const int ix, const int iy,
        VEC3 &__restrict__ dir,
        VEC3 &__restrict__ org,
        const CameraSpec &__restrict__ cam)
{
    const VEC3 v{ (float(ix) - 0.5f * float(cam.width)) / cam.fx,
                  -(float(iy) - 0.5f * float(cam.height)) / cam.fy,
                  -1.0f };
    plnoct::mv3(cam.transform, v, dir);
    dir.normalize();
    plnoct::copy3(org, cam.transform + 9);
}

template<typename IMG, typename REAL = float>
__global__ void raytrace_kernel(
        const IMG img,
        const CameraSpec cam,
        const TreeSpec tree,
        const Vector3<REAL> shift,
        const Vector3<REAL> scale)
{
    using Vec3 = Vector3<REAL>;

    const auto width = img.width();
    const auto height = img.height();
    const auto num = width * height;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num)
        return;

    auto ix = int(tid % width);
    auto iy = int(tid / width);

    REAL rgba[]{ 0, 0, 0, 0 };
    uint8_t rgba_ub[4]; // RGBA values in unsigned byte

    Vec3 dir, org;
    ::screen2worlddir(ix, iy, dir, org, cam);
    plnoct::transform_coords(org, shift, scale);

    rgba_ub[0] = uint8_t(255.0f * float(ix) / float(width));

    ::raytrace_single({ org, dir }, tree, scale, rgba);

    const auto back = fixed_opts::background_brightness;
    if (rgba[3] > 0) {
        const auto remain = back * (1.0f - rgba[3]);
        for (int i=0;i<3;i++)
            rgba_ub[i] = uint8_t(fminf(1.0f, rgba[i] + remain) * 255.0f);
        rgba_ub[3] = uint8_t(rgba[3] * 255.0f);
    } else  {
        rgba_ub[0] = uint8_t(back * 255.0f);
        rgba_ub[1] = uint8_t(back * 255.0f);
        rgba_ub[2] = uint8_t(back * 255.0f);
        rgba_ub[3] = 0;
    }

    img.write(ix, iy, rgba_ub);
}


// bilateral filtering
template<typename REAL = float>
__device__ inline REAL gaussian(const REAL x0, const REAL x1, const REAL sigma)
{
    return expf(-(x0 * x0 + x1 * x1) / (2.0 * sigma * sigma));
}

template<typename REAL = float>
__device__ inline REAL gaussian_CIELAB(
        const REAL *__restrict__ rgb0,
        const REAL *__restrict__ rgb1,
        const REAL sigma)
{
    REAL lab0[3], lab1[3];
#pragma unroll
    for (int i=0; i<3; ++i) {
        lab0[i] = rgb0[i];
        lab1[i] = rgb1[i];
    }
    plnoct::RGB_to_CIELAB(lab0);
    plnoct::RGB_to_CIELAB(lab1);

    REAL dist_squared = 0;
#pragma unroll
    for (int i=0; i<3; ++i) {
        auto d = lab0[i] - lab1[i];
        dist_squared += d * d;
    }

    return expf(-dist_squared / (2.0 * sigma * sigma));
}

template<int axis, typename IMG>
__global__ void separated_bilateral_single(const IMG src, const IMG dst)
{
    const auto width = src.width();
    const auto height = src.height();
    const auto num = width * height;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num)
        return;

    const auto ix = int(tid % width);
    const auto iy = int(tid / width);

    float rgb[]{ 0, 0, 0 };
    uint8_t rgba_ub[4]; // RGBA values in unsigned byte

    src.read(ix, iy, rgba_ub);
    if (rgba_ub[3] < g_opts.alpha_thresh) {
        dst.write(ix, iy, rgba_ub);
        return;
    }

    // cache the center pixel's rgba value
    const float cen_rgb[]{ float(rgba_ub[0]) / 255.0f,
                            float(rgba_ub[1]) / 255.0f,
                            float(rgba_ub[2]) / 255.0f };
    const auto cen_alpha = rgba_ub[3];

    float c[3], w_total = 0;
    const int radius = g_opts.filter_radius;

    for (int delta=-radius; delta<=radius; ++delta) {
        src.read(ix + int(axis == 0) * delta,
                 iy + int(axis != 0) * delta, rgba_ub);
#pragma unroll
        for (int i=0; i<3; ++i)
            c[i] = float(rgba_ub[i]) / 255.0f;

        const auto f = gaussian(0.0f, float(delta), g_opts.sigma_spatial);
        const auto g = gaussian_CIELAB(cen_rgb, c, g_opts.sigma_CIELAB);
#pragma unroll
        for (int i=0; i<3; ++i)
            rgb[i] += f * g * c[i];

        w_total += f * g;
    }

    rgba_ub[0] = uint8_t(fminf(1.0f, rgb[0] / w_total) * 255.0f);
    rgba_ub[1] = uint8_t(fminf(1.0f, rgb[1] / w_total) * 255.0f);
    rgba_ub[2] = uint8_t(fminf(1.0f, rgb[2] / w_total) * 255.0f);
    rgba_ub[3] = cen_alpha;

    dst.write(ix, iy, rgba_ub);
}

template<typename IMG>
inline void separated_bilateral_kernel(
        const int n_blocks,
        const int n_threads_per_block,
        cudaStream_t stream,
        const IMG &img0,
        const IMG &img1)
{
    ::separated_bilateral_single<0><<<n_blocks, n_threads_per_block, 0, stream>>>(img0, img1);
    ::separated_bilateral_single<1><<<n_blocks, n_threads_per_block, 0, stream>>>(img1, img0);
}

template<typename REAL>
__device__ inline void nn_brute_force(
        const REAL *__restrict__ point,
        size_t *__restrict__ nn_indices, /* vertex index of the projection */
        float *__restrict__ nn_dist_squared /* squared distance to the projection */)
{
    size_t amin_idcs[]{ 0, 0 }; // 1st & 2nd argmin indices
    REAL min_d2[]{ -1, -1 }; // 1st & 2nd min squared distances

    for (int l=0; l<::g_labels.num_labels; ++l) {
        REAL d2 = 0;
#pragma unroll
        for (int i=0; i<3; ++i) {
            const auto v = ::g_labels.reference_rgb[l*3+i] - point[i];
            d2 += v * v;
        }
        if (d2 < min_d2[0] || min_d2[0] < 0) {
            min_d2[1] = min_d2[0];
            amin_idcs[1] = amin_idcs[0];
            min_d2[0] = d2;
            amin_idcs[0] = l;
        } else if (d2 < min_d2[1] || min_d2[1] < 0) {
            min_d2[1] = d2;
            amin_idcs[1] = l;
        }
    }

#pragma unroll
    for (int i=0; i<2; ++i) {
        nn_indices[i] = amin_idcs[i];
        nn_dist_squared[i] = min_d2[i];
    }
}

template<typename IMG, typename REAL = float>
__global__ void soft_nn_projection_kernel(const IMG img)
{
    const auto width = img.width();
    const auto height = img.height();
    const auto num = width * height;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num)
        return;

    const auto ix = int(tid % width);
    const auto iy = int(tid / width);

    uint8_t rgba_ub[4]; // RGBA values in unsigned byte
    img.read(ix, iy, rgba_ub);

    if (rgba_ub[3] < ::g_opts.alpha_thresh)
        return;

    const REAL rgb[]{ float(rgba_ub[0]) / 255.0f,
                      float(rgba_ub[1]) / 255.0f,
                      float(rgba_ub[2]) / 255.0f };
    size_t nn_indices[2];
    float nn_dist_squared[2];
    ::nn_brute_force(rgb, nn_indices, nn_dist_squared);

    const auto delta = ::g_opts.step_delta;
    if (delta <= 0) {
        rgba_ub[0] = uint8_t(g_labels.recolored_rgb[nn_indices[0]*3+0] * 255.0f);
        rgba_ub[1] = uint8_t(g_labels.recolored_rgb[nn_indices[0]*3+1] * 255.0f);
        rgba_ub[2] = uint8_t(g_labels.recolored_rgb[nn_indices[0]*3+2] * 255.0f);
    } else {
        float u = 0;
        if (nn_dist_squared[0] > 0 && nn_dist_squared[1] >= 0) {
            const auto d0 = sqrtf(nn_dist_squared[0]);
            const auto d1 = sqrtf(nn_dist_squared[1]);
            const auto x = 0.5f * (1.0f + fmaxf(-delta, d0 - d1) / delta);
            u = x * x * (3.0f - 2.0f * x);
        }
#pragma unroll
        for (int i=0; i<3; ++i) {
            const auto c0 = ::g_labels.recolored_rgb[nn_indices[0]*3+i];
            const auto c1 = ::g_labels.recolored_rgb[nn_indices[1]*3+i];
            rgba_ub[i] = uint8_t(((1.0f - u) * c0 + u * c1) * 255.0f);
        }
    }

    img.write(ix, iy, rgba_ub);
}

template<class IMG>
__global__ void composite_kernel(const IMG img)
{
    const auto width = img.width();
    const auto height = img.height();
    const auto num = width * height;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num)
        return;

    const auto ix = int(tid % width);
    const auto iy = int(tid / width);

    uint8_t rgba_ub[4];
    img.read(ix, iy, rgba_ub);
    rgba_ub[3] = 255;
    img.write(ix, iy, rgba_ub);
}

} // namespace

namespace plnoct {

void render_kernel(CameraSpec &&cam,
                   TreeSpec &&tree,
                   const float *shift,
                   const float *scale,
                   Palette &palette,
                   const RenderOptions &opts,
                   cudaArray_t CA[2],
                   cudaStream_t stream)
{
    if (palette.needs_update()) {
        std::cout << "update labels" << std::endl;
        cudaMemcpyToSymbol(g_labels, &palette.labels(), sizeof(Labels));
    }

    auto width = cam.width;
    auto height = cam.height;
    int n_total = width * height;
    int n_threads_per_block = 512;
    int n_blocks = 1+(n_total-1)/n_threads_per_block;

    const ::SurfaceObjectImage img0{ CA[0], width, height };
    const ::SurfaceObjectImage img1{ CA[1], width, height };

    cudaMemcpyToSymbol(g_opts, &opts, sizeof(RenderOptions));
    ::raytrace_kernel<<<n_blocks, n_threads_per_block, 0, stream>>>(
        img0, cam, tree, Vector3<float>{ shift }, Vector3<float>{ scale }
    );
    if (opts.posterize) {
        for (int iter=0; iter<opts.num_bilateral_iter; ++iter) {
            ::separated_bilateral_kernel(n_blocks, n_threads_per_block, stream, img0, img1);
        }
        ::soft_nn_projection_kernel<<<n_blocks, n_threads_per_block, 0, stream>>>(img0);
    }

    ::composite_kernel<<<n_blocks, n_threads_per_block, 0, stream>>>(img0);
}

} // namespace plnoct