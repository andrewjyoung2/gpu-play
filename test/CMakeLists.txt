include(FetchContent)

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)

FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

#-------------------------------------------------------------------------------
set(TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.cpp
  ${CMAKE_SOURCE_DIR}/src/welcome.hpp
  ${CMAKE_SOURCE_DIR}/src/welcome.cu
  ${CMAKE_SOURCE_DIR}/src/common/math.hpp
  ${CMAKE_SOURCE_DIR}/src/common/math.cu
  ${CMAKE_SOURCE_DIR}/src/common/random_float_vector.hpp
  ${CMAKE_SOURCE_DIR}/src/common/scalar.hpp
  ${CMAKE_SOURCE_DIR}/src/common/scalar.cpp
  ${CMAKE_SOURCE_DIR}/src/cublas/cublas_wrap.hpp
  ${CMAKE_SOURCE_DIR}/src/cublas/cublas_wrap.cu
)

#-------------------------------------------------------------------------------
add_executable(unit_tests ${TEST_FILES})

target_link_libraries(unit_tests
  GTest::gtest_main
  GTest::gmock
  cublas
)
target_include_directories(unit_tests
  PRIVATE ${CMAKE_SOURCE_DIR}
  PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

