cmake_minimum_required(VERSION 3.24)

if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(RESCUE_NVCC nvcc)
    if (RESCUE_NVCC)
        set(CMAKE_CUDA_COMPILER "${RESCUE_NVCC}" CACHE FILEPATH "Path to nvcc" FORCE)
    endif()
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
endif()

project(rescue_video_analyzer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio objdetect dnn)
find_package(MPI REQUIRED)

add_executable(rescue_video_analyzer
    src/main.cpp
    src/video_io.cpp
    src/yolo_detector.cpp
    src/tracker.cpp
    src/output_writer.cpp
    src/mpi_runtime.cpp
    src/gpu_preprocess.cu
)

target_include_directories(rescue_video_analyzer PRIVATE include)

target_link_libraries(rescue_video_analyzer
    PRIVATE
        ${OpenCV_LIBS}
    MPI::MPI_CXX
    dl
)

set_target_properties(rescue_video_analyzer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

if (EXISTS "/usr/lib/wsl/lib")
    set_property(TARGET rescue_video_analyzer APPEND PROPERTY BUILD_RPATH "/usr/lib/wsl/lib")
    set_property(TARGET rescue_video_analyzer APPEND PROPERTY INSTALL_RPATH "/usr/lib/wsl/lib")
endif()
