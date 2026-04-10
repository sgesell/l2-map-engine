include(FetchContent)

# Catch2 for unit testing
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Eigen3 (header-only) — fetch if not installed system-wide
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found via find_package, fetching via FetchContent...")
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
  )
  # Disable Eigen's own tests/install to speed up configuration
  set(EIGEN_BUILD_DOC        OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING    OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING          OFF CACHE BOOL "" FORCE)
  set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
endif()
