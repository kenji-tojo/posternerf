#pragma once

#include <cassert>
#include <cmath>
#include <type_traits>

#include <cuda_runtime.h>

#define SH_BASIS_DIM_MAX 25


namespace plnoct {

namespace {

template<typename REAL>
struct Vector3 {
public:
    using value_type = REAL;

    __host__ __device__
    Vector3() = default;

    template<typename VEC3>
    __host__ __device__
    explicit Vector3(const VEC3 &other) : m_xyz{ other[0], other[1], other[2] } {
        static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    }

    __host__ __device__
    explicit Vector3(const REAL *other) : m_xyz{ other[0], other[1], other[2] } { }

    __host__ __device__
    Vector3(REAL x, REAL y, REAL z) : m_xyz{ x, y, z } { }

    __host__ __device__
    constexpr REAL operator[](size_t i) const {
        assert(i < 3);
        return m_xyz[i];
    }

    __host__ __device__
    inline REAL &operator[](size_t i) {
        assert(i < 3);
        return m_xyz[i];
    }

    __host__ __device__
    inline void operator*=(const Vector3<REAL> &other) {
        // not entirely sure why, but this eliminates diff from the original code
        this->operator[](0) *= other[0];
        this->operator[](1) *= other[1];
        this->operator[](2) *= other[2];
    }

    __host__ __device__
    inline void operator+=(const Vector3<REAL> &other) {
        m_xyz[0] += other[0];
        m_xyz[1] += other[1];
        m_xyz[2] += other[2];
    }

    __host__ __device__
    inline void operator*=(REAL s) {
        m_xyz[0] *= s;
        m_xyz[1] *= s;
        m_xyz[2] *= s;
    }

    __host__ __device__
    inline REAL norm() const {
        return sqrtf(m_xyz[0] * m_xyz[0] + m_xyz[1] * m_xyz[1] + m_xyz[2] * m_xyz[2]);
    }

    __host__ __device__
    inline void normalize() {
        auto invnorm = static_cast<REAL>(1.0f / norm());
        m_xyz[0] *= invnorm;
        m_xyz[1] *= invnorm;
        m_xyz[2] *= invnorm;
    }

private:
    REAL m_xyz[3];
};

template<typename VEC3>
struct Ray {
public:
    VEC3 org;
    VEC3 dir;
    const VEC3 vdir;

    __host__ __device__
    Ray(const VEC3 &o, const VEC3 &d) : org(o), dir(d), vdir(d) { }
};

template<typename VEC3, typename REAL>
__device__ inline void clip3(VEC3 &v, REAL lo, REAL hi) {
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    v[0] = max(lo, min(hi, v[0]));
    v[1] = max(lo, min(hi, v[1]));
    v[2] = max(lo, min(hi, v[2]));
}

template<typename VEC3, typename REAL>
__host__ __device__ inline void copy3(VEC3 &dst, const REAL *src) {
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}


template<typename VEC3>
__host__ __device__ inline void transform_coords(VEC3 &v, const VEC3 &shift, const VEC3 &scale) {
    v[0] = shift[0] + scale[0] * v[0];
    v[1] = shift[1] + scale[1] * v[1];
    v[2] = shift[2] + scale[2] * v[2];
}

template<typename REAL, typename VEC3>
__device__ inline void octree_query(
        const int32_t *__restrict__ child,
        VEC3 &__restrict__ pos,
        int64_t &__restrict__ index,
        REAL &__restrict__ cube_sz) {
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    constexpr auto N = static_cast<REAL>(2.0);
    clip3(pos, 0.0f, 1.0f - 1e-6f);
    index = 0;
    uint32_t x, y, z;
    cube_sz = N;
    for (int depth = 0; depth < 16; depth++) {
        pos *= N;
        x = static_cast<uint32_t>(floorf(pos[0]));
        y = static_cast<uint32_t>(floorf(pos[1]));
        z = static_cast<uint32_t>(floorf(pos[2]));
        pos[0] -= static_cast<REAL>(x);
        pos[1] -= static_cast<REAL>(y);
        pos[2] -= static_cast<REAL>(z);

        uint32_t morton = (x << 2) + (y << 1) + z;
        uint32_t skip = child[index + morton];
        if (skip == 0) {
            index += morton;
            return;
        }
        cube_sz *= N;
        index += (skip << 3);
    }
}

template<typename REAL, typename VEC3>
__device__ inline void eval_sh_basis(
        const VEC3 &__restrict__ dir,
        const int basis_dim,
        REAL *__restrict__ out)
{
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    // SH Coefficients from
    // https://github.com/google/spherical-harmonics
    out[0] = 0.28209479177387814;
    const REAL x = dir[0], y = dir[1], z = dir[2];
    const REAL xx = x * x, yy = y * y, zz = z * z;
    const REAL xy = x * y, yz = y * z, xz = x * z;
    switch (basis_dim) {
        case 25:
            out[16] = 2.5033429417967046 * xy * (xx - yy);
            out[17] = -1.7701307697799304 * yz * (3 * xx - yy);
            out[18] = 0.9461746957575601 * xy * (7 * zz - 1.f);
            out[19] = -0.6690465435572892 * yz * (7 * zz - 3.f);
            out[20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3);
            out[21] = -0.6690465435572892 * xz * (7 * zz - 3);
            out[22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1.f);
            out[23] = -1.7701307697799304 * xz * (xx - 3 * yy);
            out[24] = 0.6258357354491761 *
                      (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
            [[fallthrough]];
        case 16:
            out[9] = -0.5900435899266435 * y * (3 * xx - yy);
            out[10] = 2.890611442640554 * xy * z;
            out[11] = -0.4570457994644658 * y * (4 * zz - xx - yy);
            out[12] =
                    0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
            out[13] = -0.4570457994644658 * x * (4 * zz - xx - yy);
            out[14] = 1.445305721320277 * z * (xx - yy);
            out[15] = -0.5900435899266435 * x * (xx - 3 * yy);
            [[fallthrough]];
        case 9:
            out[4] = 1.0925484305920792 * xy;
            out[5] = -1.0925484305920792 * yz;
            out[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
            out[7] = -1.0925484305920792 * xz;
            out[8] = 0.5462742152960396 * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -0.4886025119029199 * y;
            out[2] = 0.4886025119029199 * z;
            out[3] = -0.4886025119029199 * x;
        default:
            break;
    }
}

template<typename REAL, typename VEC3>
__device__ static inline void itc_ray_aabb_world(
        const VEC3 &__restrict__ org,
        const VEC3 &__restrict__ invdir,
        REAL &__restrict__ tmin,
        REAL &__restrict__ tmax)
{
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    REAL t1, t2;
    tmin = 0.0;
    tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = (1e-6 - org[i]) * invdir[i];
        t2 = (1.0f - 1e-6 - org[i]) * invdir[i];
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }
}

template<typename REAL, typename VEC3>
__device__ inline REAL itc_ray_aabb_unit(
        const VEC3 &__restrict__ org,
        const VEC3 &__restrict__ invdir)
{
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    REAL t1, t2;
    REAL tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = -org[i] * invdir[i];
        t2 = t1 + invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
    return tmax;
}

template<typename REAL, typename VEC3>
__device__ inline void mv3(
        const REAL *__restrict__ m,
        const VEC3 &__restrict__ v,
        VEC3 &__restrict__ out)
{
    static_assert(std::is_same<REAL, typename VEC3::value_type>::value, "");
    out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
    out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
    out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
}

// color space conversion code adapted from
// https://github.com/ThunderStruct/Color-Utilities/blob/master/ColorUtils.cpp
template<typename REAL>
__host__ __device__ inline void RGB_to_XYZ(REAL c[3]) {
    REAL x, y, z, r, g, b;

//    r = c[0] / 255.0; g = c[1] / 255.0; b = c[2] / 255.0;
    r = c[0];
    g = c[1];
    b = c[2];

    if (r > 0.04045)
        r = powf(((r + 0.055) / 1.055), 2.4);
    else r /= 12.92;

    if (g > 0.04045)
        g = powf(((g + 0.055) / 1.055), 2.4);
    else g /= 12.92;

    if (b > 0.04045)
        b = powf(((b + 0.055) / 1.055), 2.4);
    else b /= 12.92;

    r *= 100.0;
    g *= 100.0;
    b *= 100.0;

    // Calibration for observer @2° with illumination = D65
    x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    c[0] = x;
    c[1] = y;
    c[2] = z;
}

template<typename REAL>
__host__ __device__ inline void XYZ_to_CIELAB(REAL c[3]) {
    REAL x, y, z, l, a, b;
    const REAL refX = 95.047, refY = 100.0, refZ = 108.883;

    // References set at calibration for observer @2° with illumination = D65
    x = c[0] / refX;
    y = c[1] / refY;
    z = c[2] / refZ;

    if (x > 0.008856)
        x = powf(x, 1.0 / 3.0);
    else x = (7.787 * x) + (16.0 / 116.0);

    if (y > 0.008856)
        y = powf(y, 1.0 / 3.0);
    else y = (7.787 * y) + (16.0 / 116.0);

    if (z > 0.008856)
        z = powf(z, 1.0 / 3.0);
    else z = (7.787 * z) + (16.0 / 116.0);

    l = 116.0 * y - 16.0;
    a = 500.0 * (x - y);
    b = 200.0 * (y - z);

    c[0] = l;
    c[1] = a;
    c[2] = b;
}

template<typename REAL>
__host__ __device__ inline void RGB_to_CIELAB(REAL c[3]) {
    RGB_to_XYZ<REAL>(c);
    XYZ_to_CIELAB<REAL>(c);
}

} // namespace

} // namespace plnoct