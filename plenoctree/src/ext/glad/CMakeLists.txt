cmake_minimum_required(VERSION 3.20)
project(glad LANGUAGES C)

add_library(glad STATIC src/glad.c)
target_include_directories(glad PRIVATE include)
target_compile_options(glad PRIVATE -fPIC)
