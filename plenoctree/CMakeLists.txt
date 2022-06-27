cmake_minimum_required(VERSION 3.20)
project(plenoctree LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL)
add_subdirectory(src/ext/glfw)
target_compile_options(glfw PRIVATE -fPIC)
add_subdirectory(src/ext/glad)

set(IMGUI_DIR ${CMAKE_CURRENT_LIST_DIR}/src/ext/imgui)
file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/*.cpp)

add_subdirectory(src/ext/pybind11)

if (NOT MSVC)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/_bin)
endif()

pybind11_add_module(plenoctree
    pybind.cpp
    src/octree.cpp
    src/viewer/viewer.cpp
    src/viewer/viewer.cu
    src/viewer/camera.cpp
    src/viewer/palette.cu
    src/sampling/sampling.cu
    ${IMGUI_SOURCES}
)

if (MSVC)
    set_target_properties(plenoctree PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/_bin)
    set_target_properties(plenoctree PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_LIST_DIR}/_bin)
    set_target_properties(plenoctree PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_LIST_DIR}/_bin)
endif()

target_include_directories(plenoctree PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src
    src/ext/glfw/include
    src/ext/glad/include
    src/ext/glm
    src/ext/stb
    ${IMGUI_DIR}
)

target_link_libraries(plenoctree PRIVATE glfw glad ${OPENGL_LIBRARIES})
