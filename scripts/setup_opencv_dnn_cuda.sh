#!/usr/bin/env bash
set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_BASE_DIR="${OPENCV_SRC_BASE_DIR:-$HOME/src}"
BUILD_BASE_DIR="${OPENCV_BUILD_BASE_DIR:-$HOME/build}"
INSTALL_PREFIX="${OPENCV_INSTALL_PREFIX:-/usr/local}"

OPENCV_SRC_DIR="${SRC_BASE_DIR}/opencv-${OPENCV_VERSION}"
OPENCV_CONTRIB_DIR="${SRC_BASE_DIR}/opencv_contrib-${OPENCV_VERSION}"
OPENCV_BUILD_DIR="${BUILD_BASE_DIR}/opencv-${OPENCV_VERSION}-build-cuda"

JOBS="${JOBS:-$(nproc)}"

log() {
    echo "[setup-opencv-cuda] $*"
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

ensure_nvcc_in_path() {
    if command -v nvcc >/dev/null 2>&1; then
        return
    fi

    local candidate
    for candidate in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        if [[ -x "${candidate}" ]]; then
            export PATH="$(dirname "${candidate}"):${PATH}"
            log "nvcc found at ${candidate} (added to PATH for this run)"
            return
        fi
    done
}

has_cudnn_headers() {
    [[ -f /usr/include/cudnn.h ]] ||
    [[ -f /usr/local/cuda/include/cudnn.h ]] ||
    [[ -f /usr/include/x86_64-linux-gnu/cudnn_version.h ]] ||
    [[ -f /usr/local/cuda/include/cudnn_version.h ]]
}

has_cudnn_libs() {
    ldconfig -p 2>/dev/null | grep -qi libcudnn
}

compiler_major() {
    local compiler="$1"
    "${compiler}" -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1
}

select_host_compilers() {
    if [[ -n "${HOST_C_COMPILER:-}" && -n "${HOST_CXX_COMPILER:-}" ]]; then
        echo "${HOST_C_COMPILER};${HOST_CXX_COMPILER}"
        return
    fi

    if [[ -x /usr/bin/gcc-12 && -x /usr/bin/g++-12 ]]; then
        echo "/usr/bin/gcc-12;/usr/bin/g++-12"
        return
    fi

    if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
        echo "$(command -v gcc-12);$(command -v g++-12)"
        return
    fi

    if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
        echo "$(command -v gcc);$(command -v g++)"
        return
    fi

    echo ";"
}

detect_cuda_arch_bin() {
    if [[ -n "${CUDA_ARCH_BIN:-}" ]]; then
        echo "${CUDA_ARCH_BIN}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
        if [[ -n "${cap}" ]]; then
            echo "${cap}"
            return
        fi
    fi

    if [[ -f /proc/driver/nvidia/gpus/0000:01:00.0/information ]]; then
        log "Warning: could not detect CUDA compute capability via nvidia-smi."
    fi

    echo "8.9"
}

log "Checking required tools"
ensure_nvcc_in_path
need_cmd git
need_cmd cmake
need_cmd make
need_cmd nvcc

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "Warning: nvidia-smi not found inside WSL. CUDA runtime may not be accessible."
fi

if ! has_cudnn_headers || ! has_cudnn_libs; then
    cat >&2 <<'EOF'
cuDNN not detected.

This project needs OpenCV DNN CUDA support, which requires cuDNN headers and libs.
Install cuDNN for your CUDA version first, then re-run this script.

Quick checks:
  - ls /usr/local/cuda/include | grep cudnn
  - ldconfig -p | grep libcudnn
EOF
    exit 1
fi

log "Installing build dependencies"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gcc-12 \
    g++-12 \
    pkg-config \
    git \
    cmake \
    ninja-build \
    python3-numpy \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenexr-dev \
    libtbb-dev \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libprotobuf-dev \
    protobuf-compiler

mkdir -p "${SRC_BASE_DIR}" "${BUILD_BASE_DIR}"

HOST_COMPILERS="$(select_host_compilers)"
HOST_C_COMPILER_PATH="${HOST_COMPILERS%%;*}"
HOST_CXX_COMPILER_PATH="${HOST_COMPILERS##*;}"

if [[ -z "${HOST_C_COMPILER_PATH}" || -z "${HOST_CXX_COMPILER_PATH}" ]]; then
    echo "Could not resolve host C/C++ compilers for CUDA build." >&2
    exit 1
fi

HOST_CXX_MAJOR="$(compiler_major "${HOST_CXX_COMPILER_PATH}")"
if [[ -z "${HOST_CXX_MAJOR}" ]]; then
    echo "Could not detect compiler version for ${HOST_CXX_COMPILER_PATH}." >&2
    exit 1
fi

if (( HOST_CXX_MAJOR > 12 )); then
    echo "CUDA 12.0 does not support host compiler major ${HOST_CXX_MAJOR}." >&2
    echo "Install gcc-12/g++-12 or export HOST_C_COMPILER and HOST_CXX_COMPILER." >&2
    exit 1
fi

log "Using host compilers: C=${HOST_C_COMPILER_PATH}, CXX=${HOST_CXX_COMPILER_PATH}"

if [[ ! -d "${OPENCV_SRC_DIR}/.git" ]]; then
    log "Cloning OpenCV ${OPENCV_VERSION}"
    git clone --branch "${OPENCV_VERSION}" --depth 1 https://github.com/opencv/opencv.git "${OPENCV_SRC_DIR}"
else
    log "OpenCV source already exists: ${OPENCV_SRC_DIR}"
fi

if [[ ! -d "${OPENCV_CONTRIB_DIR}/.git" ]]; then
    log "Cloning opencv_contrib ${OPENCV_VERSION}"
    git clone --branch "${OPENCV_VERSION}" --depth 1 https://github.com/opencv/opencv_contrib.git "${OPENCV_CONTRIB_DIR}"
else
    log "opencv_contrib source already exists: ${OPENCV_CONTRIB_DIR}"
fi

if [[ -f "${OPENCV_BUILD_DIR}/CMakeCache.txt" ]]; then
    CACHED_CXX="$(grep -E '^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=' "${OPENCV_BUILD_DIR}/CMakeCache.txt" | head -n1 | cut -d= -f2- || true)"
    if [[ -n "${CACHED_CXX}" && "${CACHED_CXX}" != "${HOST_CXX_COMPILER_PATH}" ]]; then
        log "Compiler changed (${CACHED_CXX} -> ${HOST_CXX_COMPILER_PATH}), cleaning cached OpenCV build dir"
        rm -rf "${OPENCV_BUILD_DIR}"
    fi
fi

mkdir -p "${OPENCV_BUILD_DIR}"

CUDA_ARCH_ARG=()
CUDA_ARCH_BIN_VALUE="$(detect_cuda_arch_bin)"
CUDA_ARCH_ARG=(-D CUDA_ARCH_BIN="${CUDA_ARCH_BIN_VALUE}")
log "Using CUDA_ARCH_BIN=${CUDA_ARCH_BIN_VALUE}"

log "Configuring OpenCV with CUDA DNN"
cmake -S "${OPENCV_SRC_DIR}" -B "${OPENCV_BUILD_DIR}" -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER="${HOST_C_COMPILER_PATH}" \
    -D CMAKE_CXX_COMPILER="${HOST_CXX_COMPILER_PATH}" \
    -D CUDA_HOST_COMPILER="${HOST_CXX_COMPILER_PATH}" \
    -D CMAKE_CUDA_HOST_COMPILER="${HOST_CXX_COMPILER_PATH}" \
    -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -D OPENCV_EXTRA_MODULES_PATH="${OPENCV_CONTRIB_DIR}/modules" \
    -D WITH_CUDA=ON \
    -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_LIST=core,imgproc,imgcodecs,videoio,objdetect,dnn,highgui,cudev \
    "${CUDA_ARCH_ARG[@]}"

log "Building OpenCV (jobs=${JOBS})"
cmake --build "${OPENCV_BUILD_DIR}" --parallel "${JOBS}"

log "Installing OpenCV"
sudo cmake --install "${OPENCV_BUILD_DIR}"
sudo ldconfig

if command -v opencv_version >/dev/null 2>&1; then
    log "OpenCV build info (CUDA/cuDNN lines):"
    opencv_version --verbose | grep -E "NVIDIA CUDA|cuDNN|To be built|Disabled" || true
fi

log "Rebuilding project"
cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" \
    -D OpenCV_DIR="${INSTALL_PREFIX}/lib/cmake/opencv4" \
    -D CMAKE_CUDA_COMPILER=/usr/bin/nvcc
cmake --build "${PROJECT_ROOT}/build" -j

cat <<'EOF'

Done.

Validation commands:
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 2 ./build/rescue_video_analyzer \
    --input videoset.mp4 \
    --output-dir output_videoset \
    --batch-size 8 \
    --resize-width 640

    LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 2 ./build/rescue_video_analyzer \
    --input videoset2.mp4 \
    --output-dir output_videoset2 \
    --batch-size 8 \
    --resize-width 640
EOF
