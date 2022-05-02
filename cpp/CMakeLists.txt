cmake_minimum_required(VERSION 3.16)

project(GaussianAmplitudeComputer)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_library(gaussian_computer_dependency INTERFACE)
target_compile_features(gaussian_computer_dependency INTERFACE cxx_std_20)

# Compile options
option(ENABLE_SANITIZE "Enable sanitize address" OFF)
option(ENABLE_WARNINGS "Enable warnings" ON)

# Other build options
option(BUILD_TESTS "Build cpp tests" OFF)

if (ENABLE_SANITIZE)
    target_compile_options(gaussian_computer_dependency INTERFACE "-fsanitize=address")
    target_link_options(gaussian_computer_dependency INTERFACE "-fsanitize=address")
endif()

if(ENABLE_WARNINGS)
    message(STATUS "ENABLE_WARNINGS is ON.")
    if(MSVC)
        target_compile_options(gaussian_computer_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>)
    else()
        target_compile_options(gaussian_computer_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
    endif()
else()
    message(STATUS "ENABLE_WARNINGS is OFF.")
endif()

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native.")
    target_compile_options(gaussian_computer_dependency INTERFACE -march=native)
else()
    message(STATUS "ENABLE_NATIVE is OFF.")
endif()

if(ENABLE_OPENMP)
     message(STATUS "ENABLE_OPENMP is ON.")
    find_package(OpenMP)

    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "OpenMP is enabled but not found.\n"
            "Install OpenMP or set ENABLE_OPENMP OFF.")
    endif()

    target_link_libraries(gaussian_computer_dependency INTERFACE OpenMP::OpenMP_CXX)
else()
    message(STATUS "ENABLE_OPENMP is OFF.")
endif()



add_subdirectory(src)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
