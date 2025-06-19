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
  ${CMAKE_SOURCE_DIR}/src/common/assert.hpp
  ${CMAKE_SOURCE_DIR}/src/common/file_io.cpp
  ${CMAKE_SOURCE_DIR}/src/common/file_io.hpp
  ${CMAKE_SOURCE_DIR}/src/common/math.cu
  ${CMAKE_SOURCE_DIR}/src/common/math.hpp
  ${CMAKE_SOURCE_DIR}/src/common/matrix.hpp
  ${CMAKE_SOURCE_DIR}/src/common/random_float_vector.hpp
  ${CMAKE_SOURCE_DIR}/src/common/scalar.cpp
  ${CMAKE_SOURCE_DIR}/src/common/scalar.hpp
  ${CMAKE_SOURCE_DIR}/src/common/vector.hpp
  ${CMAKE_SOURCE_DIR}/src/cublas/cublas_wrap.cu
  ${CMAKE_SOURCE_DIR}/src/cublas/cublas_wrap.hpp
  ${CMAKE_SOURCE_DIR}/src/em_alg/posterior.cpp
  ${CMAKE_SOURCE_DIR}/src/em_alg/posterior.cu
  ${CMAKE_SOURCE_DIR}/src/em_alg/posterior.hpp
  ${CMAKE_SOURCE_DIR}/src/em_alg/covar_est.cpp
  ${CMAKE_SOURCE_DIR}/src/em_alg/covar_est.hpp
  ${CMAKE_SOURCE_DIR}/src/em_alg/mean_est.cpp
  ${CMAKE_SOURCE_DIR}/src/em_alg/mean_est.cu
  ${CMAKE_SOURCE_DIR}/src/em_alg/mean_est.hpp
  ${CMAKE_SOURCE_DIR}/src/welcome.cu
  ${CMAKE_SOURCE_DIR}/src/welcome.hpp
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

