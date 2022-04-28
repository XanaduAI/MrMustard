cmake_minimum_required(VERSION 3.14)

project(GaussianAmplitudeComputer_tests)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()


Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.9
)

FetchContent_MakeAvailable(Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)


################################################################################
# Define compile options and library
################################################################################

set(TEST_SOURCES runner.cpp test_utils.cpp)

add_executable(runner ${TEST_SOURCES})
target_link_libraries(runner PRIVATE Catch2::Catch2 gaussian_computer_src)
