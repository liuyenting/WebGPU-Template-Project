cmake_minimum_required(VERSION 3.9)

project(
    ECE408
    LANGUAGES CUDA CXX
)
#####
# compile environment sanity check
#####
# we need CUDA for sure
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

#####
# WebGPU
#####
add_subdirectory(libwb)
# libwb somehow does not have its own include dir
include_directories(libwb)

#####
# different MPs
#####
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

option(BUILD_MP0 "Build MP0" ON)
option(BUILD_MP1 "Build MP1" ON)

if(BUILD_MP0)
    add_subdirectory(mp0)
endif()

if(BUILD_MP1)
    add_subdirectory(mp1)
endif()
