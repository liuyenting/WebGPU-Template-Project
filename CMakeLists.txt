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
option(BUILD_MP2 "Build MP2" ON)
option(BUILD_MP3 "Build MP3" ON)
option(BUILD_MP4 "Build MP4" ON)
option(BUILD_MP5_1 "Build MP5.1" ON)
option(BUILD_MP5_2 "Build MP5.2" ON)
option(BUILD_MP6 "Build MP6" ON)

if(BUILD_MP0)
    add_subdirectory(mp0)
endif()

if(BUILD_MP1)
    add_subdirectory(mp1)
endif()

if(BUILD_MP2)
    add_subdirectory(mp2)
endif()

if(BUILD_MP3)
    add_subdirectory(mp3)
endif()

if(BUILD_MP4)
    add_subdirectory(mp4)
endif()

if(BUILD_MP5_1)
    add_subdirectory(mp5_1)
endif()

if(BUILD_MP5_2)
    add_subdirectory(mp5_2)
endif()

if(BUILD_MP6)
    add_subdirectory(mp6)
endif()