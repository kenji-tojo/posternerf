#pragma once

#include <cstdint>
#include <cuda_fp16.h>


namespace plnoct {

struct CameraSpec {
    int width, height;
    float fx, fy;
    float transform[12];

    CameraSpec(int _width, int _height, float _fx, float _fy, const float *_transform)
    : width(_width) , height(_height)
    , fx(_fx) , fy(_fy)
    , transform{
        _transform[0], _transform[1], _transform[2],
        _transform[3], _transform[4], _transform[5],
        _transform[6], _transform[7], _transform[8],
        _transform[9], _transform[10], _transform[11]
    }
    { }
};

struct TreeSpec{
    int data_dim;
    int basis_dim;
    const __half *data;
    const int32_t *child;
};

}