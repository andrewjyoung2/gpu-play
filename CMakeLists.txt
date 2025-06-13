cmake_minimum_required(VERSION 3.16...3.27)
project(GPU_Play LANGUAGES CXX CUDA)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic-errors")

enable_testing()

add_subdirectory(src)
add_subdirectory(test)


