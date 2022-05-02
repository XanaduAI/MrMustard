cmake_minimum_required(VERSION 3.16)

project(GaussianAmplitudeComputer_src)

add_library(gaussian_computer_lib STATIC utils.cpp)
target_include_directories(gaussian_computer_lib PUBLIC .)
target_link_libraries(gaussian_computer_lib PUBLIC gaussian_computer_dependency)
set_property(TARGET gaussian_computer_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Add pybind11
include(FetchContent)
FetchContent_Declare(pybind11
                     GIT_REPOSITORY https://github.com/pybind/pybind11.git
                     GIT_TAG        v2.9.2
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(mrmustard_cpp_math "binding.cpp")
target_link_libraries(mrmustard_cpp_math PRIVATE gaussian_computer_lib)
set_target_properties(mrmustard_cpp_math PROPERTIES CXX_VISIBILITY_PRESET hidden)
