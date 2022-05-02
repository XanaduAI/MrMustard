cmake_minimum_required(VERSION 3.14)

project(GaussianAmplitudeComputer_tests)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()


Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.0-preview5
)

FetchContent_MakeAvailable(Catch2)

# Modify `ctest` to only run the supported subset of tests.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)


################################################################################
# Define compile options and library
################################################################################

set(TEST_SOURCES test_utils.cpp test_amplitude_computer.cpp)

add_executable(runner ${TEST_SOURCES})
target_link_libraries(runner PRIVATE Catch2::Catch2WithMain gaussian_computer_src)

catch_discover_tests(runner)
