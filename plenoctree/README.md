# Radiance Sampler & Real-Time Stylized Renderer

Implementation of our radiance sampling and real-time stylizaed rendering.

This library uses the efficient data structure **PlenOctree** for volumetric radiance fields.\
The ray tracing of the PlenOctree and the OpenGL viewer logic are adapted from the [official interactive PlenOctre viewer](https://github.com/sxyu/volrend).

## Build Instructions

The library is built as a python module using [pybind11](https://github.com/pybind/pybind11). Follow the instruction at the project root for building.

## References
- Yu et al. 2021. [PlenOctrees For Real-time Rendering of Neural Radiance Fields](https://alexyu.net/plenoctrees/)