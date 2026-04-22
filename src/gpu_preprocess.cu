#include "rescue/gpu_preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace rescue {

namespace {

constexpr std::uint8_t kLetterboxPadValue = 114;

struct DeviceStats {
    unsigned long long intensity_sum;
    unsigned int edge_count;
};

void cuda_check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + call + ": " + cudaGetErrorString(error));
    }
}

void release_device_tensor(float*& device_input_tensor) noexcept {
    if (device_input_tensor != nullptr) {
        cudaFree(device_input_tensor);
        device_input_tensor = nullptr;
    }
}

__device__ float clamp_float(float value, float lower, float upper) {
    return fminf(fmaxf(value, lower), upper);
}

__device__ float read_channel(
    const std::uint8_t* input_bgr,
    int input_pitch,
    int x,
    int y,
    int channel) {
    return static_cast<float>(input_bgr[y * input_pitch + (x * 3) + channel]);
}

__device__ float bilinear_sample_channel(
    const std::uint8_t* input_bgr,
    int input_width,
    int input_height,
    int input_pitch,
    float x,
    float y,
    int channel) {
    x = clamp_float(x, 0.0f, static_cast<float>(input_width - 1));
    y = clamp_float(y, 0.0f, static_cast<float>(input_height - 1));

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, input_width - 1);
    const int y1 = min(y0 + 1, input_height - 1);

    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const float top =
        read_channel(input_bgr, input_pitch, x0, y0, channel) * (1.0f - dx) +
        read_channel(input_bgr, input_pitch, x1, y0, channel) * dx;
    const float bottom =
        read_channel(input_bgr, input_pitch, x0, y1, channel) * (1.0f - dx) +
        read_channel(input_bgr, input_pitch, x1, y1, channel) * dx;

    return top * (1.0f - dy) + bottom * dy;
}

__global__ void letterbox_to_tensor_and_gray_kernel(
    const std::uint8_t* input_bgr,
    int input_width,
    int input_height,
    int input_pitch,
    float* output_tensor,
    std::uint8_t* output_gray,
    int output_width,
    int output_height,
    int resized_width,
    int resized_height,
    int pad_x,
    int pad_y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) {
        return;
    }

    float b = static_cast<float>(kLetterboxPadValue);
    float g = static_cast<float>(kLetterboxPadValue);
    float r = static_cast<float>(kLetterboxPadValue);

    const bool inside_content =
        x >= pad_x && x < pad_x + resized_width &&
        y >= pad_y && y < pad_y + resized_height;

    if (inside_content) {
        const float scale_x = static_cast<float>(input_width) / static_cast<float>(resized_width);
        const float scale_y = static_cast<float>(input_height) / static_cast<float>(resized_height);

        const float src_x = (static_cast<float>(x - pad_x) + 0.5f) * scale_x - 0.5f;
        const float src_y = (static_cast<float>(y - pad_y) + 0.5f) * scale_y - 0.5f;

        b = bilinear_sample_channel(input_bgr, input_width, input_height, input_pitch, src_x, src_y, 0);
        g = bilinear_sample_channel(input_bgr, input_width, input_height, input_pitch, src_x, src_y, 1);
        r = bilinear_sample_channel(input_bgr, input_width, input_height, input_pitch, src_x, src_y, 2);
    }

    const int pixel_index = y * output_width + x;
    const int plane_size = output_width * output_height;
    const float scale = 1.0f / 255.0f;

    output_tensor[pixel_index] = r * scale;
    output_tensor[plane_size + pixel_index] = g * scale;
    output_tensor[(plane_size * 2) + pixel_index] = b * scale;
    output_gray[pixel_index] = static_cast<std::uint8_t>(0.114f * b + 0.587f * g + 0.299f * r + 0.5f);
}

__global__ void stats_kernel(
    const std::uint8_t* gray,
    int width,
    int height,
    int edge_threshold,
    int content_left,
    int content_top,
    int content_right,
    int content_bottom,
    DeviceStats* stats) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    if (x < content_left || x >= content_right || y < content_top || y >= content_bottom) {
        return;
    }

    const int idx = y * width + x;
    const std::uint8_t center = gray[idx];
    atomicAdd(&stats->intensity_sum, static_cast<unsigned long long>(center));

    if (x <= content_left || y <= content_top || x >= content_right - 1 || y >= content_bottom - 1) {
        return;
    }

    const int gx =
        -gray[(y - 1) * width + (x - 1)] - 2 * gray[y * width + (x - 1)] - gray[(y + 1) * width + (x - 1)] +
        gray[(y - 1) * width + (x + 1)] + 2 * gray[y * width + (x + 1)] + gray[(y + 1) * width + (x + 1)];

    const int gy =
        -gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
        gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];

    const int magnitude = abs(gx) + abs(gy);
    if (magnitude > edge_threshold) {
        atomicAdd(&stats->edge_count, 1u);
    }
}

}  // namespace

GpuFrameArtifacts::~GpuFrameArtifacts() {
    release_device_tensor(device_input_tensor);
}

GpuFrameArtifacts::GpuFrameArtifacts(GpuFrameArtifacts&& other) noexcept {
    *this = std::move(other);
}

GpuFrameArtifacts& GpuFrameArtifacts::operator=(GpuFrameArtifacts&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release_device_tensor(device_input_tensor);

    device_input_tensor = other.device_input_tensor;
    input_tensor_bytes = other.input_tensor_bytes;
    original_frame_size = other.original_frame_size;
    model_input_size = other.model_input_size;
    resize_ratio = other.resize_ratio;
    pad_x = other.pad_x;
    pad_y = other.pad_y;
    mean_intensity = other.mean_intensity;
    edge_density = other.edge_density;

    other.device_input_tensor = nullptr;
    other.input_tensor_bytes = 0;
    other.original_frame_size = {};
    other.model_input_size = {};
    other.resize_ratio = 1.0;
    other.pad_x = 0;
    other.pad_y = 0;
    other.mean_intensity = 0.0;
    other.edge_density = 0.0;

    return *this;
}

GpuFrameArtifacts preprocess_frame_cuda(
    const cv::Mat& bgr_frame,
    const cv::Size& model_input_size,
    int edge_threshold) {
    if (bgr_frame.empty()) {
        throw std::runtime_error("CUDA preprocessing received an empty frame.");
    }
    if (bgr_frame.type() != CV_8UC3) {
        throw std::runtime_error("CUDA preprocessing expects CV_8UC3 input.");
    }
    if (model_input_size.width <= 0 || model_input_size.height <= 0) {
        throw std::runtime_error("CUDA preprocessing received an invalid model input size.");
    }

    const cv::Mat contiguous_frame = bgr_frame.isContinuous() ? bgr_frame : bgr_frame.clone();

    const double resize_ratio = std::min(
        static_cast<double>(model_input_size.width) / static_cast<double>(contiguous_frame.cols),
        static_cast<double>(model_input_size.height) / static_cast<double>(contiguous_frame.rows));

    const int resized_width = std::max(1, static_cast<int>(std::lround(contiguous_frame.cols * resize_ratio)));
    const int resized_height = std::max(1, static_cast<int>(std::lround(contiguous_frame.rows * resize_ratio)));
    const int pad_x = (model_input_size.width - resized_width) / 2;
    const int pad_y = (model_input_size.height - resized_height) / 2;

    const std::size_t input_bytes =
        static_cast<std::size_t>(contiguous_frame.step) * static_cast<std::size_t>(contiguous_frame.rows);
    const std::size_t gray_bytes =
        static_cast<std::size_t>(model_input_size.width) * static_cast<std::size_t>(model_input_size.height);
    const std::size_t input_tensor_bytes =
        static_cast<std::size_t>(model_input_size.width) *
        static_cast<std::size_t>(model_input_size.height) *
        3u *
        sizeof(float);

    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_gray = nullptr;
    float* d_input_tensor = nullptr;
    DeviceStats* d_stats = nullptr;

    auto cleanup = [&]() noexcept {
        if (d_input != nullptr) {
            cudaFree(d_input);
        }
        if (d_gray != nullptr) {
            cudaFree(d_gray);
        }
        if (d_input_tensor != nullptr) {
            cudaFree(d_input_tensor);
        }
        if (d_stats != nullptr) {
            cudaFree(d_stats);
        }
    };

    try {
        cuda_check(cudaMalloc(&d_input, input_bytes), "cudaMalloc(d_input)");
        cuda_check(cudaMalloc(&d_gray, gray_bytes), "cudaMalloc(d_gray)");
        cuda_check(cudaMalloc(&d_input_tensor, input_tensor_bytes), "cudaMalloc(d_input_tensor)");
        cuda_check(cudaMalloc(&d_stats, sizeof(DeviceStats)), "cudaMalloc(d_stats)");

        DeviceStats zero_stats{0ull, 0u};
        cuda_check(
            cudaMemcpy(d_input, contiguous_frame.data, input_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(input)");
        cuda_check(cudaMemcpy(d_stats, &zero_stats, sizeof(DeviceStats), cudaMemcpyHostToDevice), "cudaMemcpy(stats)");

        const dim3 block(16, 16);
        const dim3 grid(
            (model_input_size.width + block.x - 1) / block.x,
            (model_input_size.height + block.y - 1) / block.y);

        letterbox_to_tensor_and_gray_kernel<<<grid, block>>>(
            d_input,
            contiguous_frame.cols,
            contiguous_frame.rows,
            static_cast<int>(contiguous_frame.step),
            d_input_tensor,
            d_gray,
            model_input_size.width,
            model_input_size.height,
            resized_width,
            resized_height,
            pad_x,
            pad_y);
        cuda_check(cudaGetLastError(), "letterbox_to_tensor_and_gray_kernel launch");

        stats_kernel<<<grid, block>>>(
            d_gray,
            model_input_size.width,
            model_input_size.height,
            edge_threshold,
            pad_x,
            pad_y,
            pad_x + resized_width,
            pad_y + resized_height,
            d_stats);
        cuda_check(cudaGetLastError(), "stats_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        DeviceStats host_stats{};
        cuda_check(cudaMemcpy(&host_stats, d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost), "cudaMemcpy(stats)");

        GpuFrameArtifacts artifacts;
        artifacts.device_input_tensor = d_input_tensor;
        artifacts.input_tensor_bytes = input_tensor_bytes;
        artifacts.original_frame_size = contiguous_frame.size();
        artifacts.model_input_size = model_input_size;
        artifacts.resize_ratio = resize_ratio;
        artifacts.pad_x = pad_x;
        artifacts.pad_y = pad_y;

        const double content_pixels = static_cast<double>(resized_width) * static_cast<double>(resized_height);
        artifacts.mean_intensity = content_pixels > 0.0 ? static_cast<double>(host_stats.intensity_sum) / content_pixels : 0.0;
        artifacts.edge_density = content_pixels > 0.0 ? static_cast<double>(host_stats.edge_count) / content_pixels : 0.0;

        d_input_tensor = nullptr;
        cleanup();
        return artifacts;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace rescue
