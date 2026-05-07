cmake_minimum_required(VERSION 3.24)

if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(RESCUE_NVCC nvcc)
    if (RESCUE_NVCC)
        set(CMAKE_CUDA_COMPILER "${RESCUE_NVCC}" CACHE FILEPATH "Path to nvcc" FORCE)
    endif()
endif()

project(rescue_video_analyzer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio objdetect dnn)
find_package(MPI REQUIRED)

option(RESCUE_ENABLE_CUDA "Enable CUDA backend" ON)

if (RESCUE_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
    endif()
    add_compile_definitions(RESCUE_ENABLE_CUDA=1)
endif()

set(RESCUE_SOURCES
    src/main.cpp
    src/video_io.cpp
    src/mpi_runtime.cpp
    src/output_writer.cpp
    src/tracker.cpp
    src/gpu_preprocess_opencl.cpp
    src/yolo_detector_opencl.cpp
)

if (RESCUE_ENABLE_CUDA)
    list(APPEND RESCUE_SOURCES
        src/yolo_detector_cuda.cpp
        src/gpu_preprocess.cu
    )
endif()

add_executable(rescue_video_analyzer ${RESCUE_SOURCES})

target_include_directories(rescue_video_analyzer PRIVATE include)

target_link_libraries(rescue_video_analyzer
    PRIVATE
        ${OpenCV_LIBS}
    MPI::MPI_CXX
    dl
)

if (RESCUE_ENABLE_CUDA)
    set_target_properties(rescue_video_analyzer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

if (EXISTS "/usr/lib/wsl/lib")
    set_property(TARGET rescue_video_analyzer APPEND PROPERTY BUILD_RPATH "/usr/lib/wsl/lib")
    set_property(TARGET rescue_video_analyzer APPEND PROPERTY INSTALL_RPATH "/usr/lib/wsl/lib")
endif()
