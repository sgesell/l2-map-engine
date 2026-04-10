cmake_minimum_required(VERSION 3.18)
project(L2MapEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies (Eigen3, Catch2, pybind11 — with FetchContent fallbacks)
include(cmake/FetchDependencies.cmake)

find_package(OpenMP)   # Optional — warn if missing, don't hard-fail

option(L2MAP_BUILD_PYTHON "Build Python bindings" ON)
option(L2MAP_BUILD_TESTS  "Build tests"           ON)

# Core library — STATIC avoids Windows DLL path issues during test discovery
add_library(l2map STATIC
  src/element_library.cpp
  src/element_library_3d.cpp
  src/mesh.cpp
  src/bvh.cpp
  src/basis_builder.cpp
  src/basis_builder_3d.cpp
  src/polygon_clipper.cpp
  src/poly_integrator.cpp
  src/mapping_engine.cpp
  src/mapping_engine_3d.cpp
  src/polyhedron_clipper.cpp
  src/poly_integrator_3d.cpp
  src/mapping_engine_3d_exact.cpp
  src/parallel_executor.cpp
  src/io.cpp
)

target_include_directories(l2map PUBLIC include)
target_link_libraries(l2map PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
  target_link_libraries(l2map PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found — building single-threaded fallback")
  # Stub out omp_get_max_threads for the parallel_executor
  target_compile_definitions(l2map PRIVATE L2MAP_NO_OPENMP)
endif()

# Compiler warnings — use generator expressions to support MSVC and GCC/Clang
target_compile_options(l2map PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4127 /wd4244>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
)

# Release optimisations (GCC/Clang only — MSVC handles this via /O2 in Release)
target_compile_options(l2map PRIVATE
  $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native>
)

if(L2MAP_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(L2MAP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
