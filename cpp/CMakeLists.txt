cmake_minimum_required(VERSION 3.16)

project(GaussianAmplitudeComputer)

add_library(gaussian_computer_dependency INTERFACE)
target_compile_features(gaussian_computer_dependency INTERFACE cxx_std_20)


target_compile_options(gaussian_computer_dependency INTERFACE "-fsanitize=address;-O0")
target_link_options(gaussian_computer_dependency INTERFACE "-fsanitize=address")

add_subdirectory(src)
add_subdirectory(tests)
