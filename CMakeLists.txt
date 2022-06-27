cmake_minimum_required(VERSION 3.20)
project(poster_nerf LANGUAGES C CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(plenoctree)

add_custom_target(poster_nerf)
add_dependencies(poster_nerf plenoctree)