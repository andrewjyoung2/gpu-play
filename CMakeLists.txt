cmake_minimum_required(VERSION 3.16...3.27)
project(GPU_Play LANGUAGES CXX CUDA)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic-errors")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets")

enable_testing()

add_subdirectory(test)

