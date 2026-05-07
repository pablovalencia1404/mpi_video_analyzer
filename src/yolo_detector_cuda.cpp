#include "rescue/detector.hpp"
#include "rescue/gpu_preprocess_cuda.hpp"

#include "rescue/onnxruntime_c_api_minimal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>

namespace rescue {

namespace {

struct LetterboxInfo {
    double ratio = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};

constexpr std::int32_t kPersonClassId = 0;

std::string read_env_string(const char* name) {
    if (name == nullptr) {
        return {};
    }

    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return {};
    }

    return value;
}

template <typename T>
struct ApiDeleter {
    using ReleaseFn = void (*)(T*);

    ReleaseFn release = nullptr;

    void operator()(T* value) const noexcept {
        if (value != nullptr && release != nullptr) {
            release(value);
        }
    }
};

template <typename T>
using ApiPtr = std::unique_ptr<T, ApiDeleter<T>>;

template <typename T>
ApiPtr<T> make_api_ptr(T* value, typename ApiDeleter<T>::ReleaseFn release) {
    ApiPtr<T> result(value);
    result.get_deleter().release = release;
    return result;
}

template <typename Fn>
Fn load_symbol(void* library, const char* symbol_name) {
    dlerror();
    void* symbol = dlsym(library, symbol_name);
    const char* error = dlerror();
    if (error != nullptr) {
        throw std::runtime_error(std::string("Failed to resolve symbol ") + symbol_name + ": " + error);
    }

    return reinterpret_cast<Fn>(symbol);
}

std::vector<std::filesystem::path> ort_library_candidates() {
    const auto root = []() {
        if (const char* root_env = std::getenv("RESCUE_PROJECT_ROOT"); root_env != nullptr && root_env[0] != '\0') {
            return std::filesystem::path(root_env);
        }

        return std::filesystem::current_path();
    }();

    return {
        root / ".venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4",
        root / ".venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so",
        root / ".venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24",
        root / ".venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4",
        root / ".venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so",
    };
}

std::vector<std::filesystem::path> venv_site_packages_roots() {
    const auto root = []() {
        if (const char* root_env = std::getenv("RESCUE_PROJECT_ROOT"); root_env != nullptr && root_env[0] != '\0') {
            return std::filesystem::path(root_env);
        }

        return std::filesystem::current_path();
    }();

    return {
        root / ".venv/lib/python3.12/site-packages",
        root / ".venv/lib64/python3.12/site-packages",
        root / ".venv/lib/python3.11/site-packages",
        root / ".venv/lib64/python3.11/site-packages",
    };
}

void preload_library(const std::filesystem::path& library_path) {
    if (!std::filesystem::exists(library_path)) {
        return;
    }

    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        throw std::runtime_error(std::string("Failed to preload ") + library_path.string() + ": " + dlerror());
    }
}

void preload_cuda_support_libraries() {
    for (const auto& root : venv_site_packages_roots()) {
        preload_library(root / "nvidia/cu13/lib/libnvJitLink.so.13");
        preload_library(root / "nvidia/cu13/lib/libnvrtc-builtins.so.13.0");
        preload_library(root / "nvidia/cu13/lib/libcublasLt.so.13");
        preload_library(root / "nvidia/cu13/lib/libcublas.so.13");
        preload_library(root / "nvidia/cu13/lib/libnvrtc.so.13");
        preload_library(root / "nvidia/cu13/lib/libcurand.so.10");
        preload_library(root / "nvidia/cu13/lib/libcufft.so.12");
        preload_library(root / "nvidia/cu13/lib/libcudart.so.13");
        preload_library(root / "nvidia/cudnn/lib/libcudnn.so.9");
    }
}

void* open_ort_library() {
    if (const char* override_path = std::getenv("RESCUE_ORT_LIBRARY"); override_path != nullptr && override_path[0] != '\0') {
        void* handle = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            return handle;
        }

        throw std::runtime_error(std::string("Failed to load ORT library from RESCUE_ORT_LIBRARY: ") + dlerror());
    }

    for (const auto& candidate : ort_library_candidates()) {
        if (!std::filesystem::exists(candidate)) {
            continue;
        }

        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            return handle;
        }
    }

    throw std::runtime_error(
        "Unable to locate libonnxruntime.so.1.24.4 in the local .venv. Set RESCUE_ORT_LIBRARY if needed.");
}

cv::Rect2d clip_box(const cv::Rect2d& box, const cv::Size& frame_size) {
    const double left = std::max(0.0, std::min(box.x, static_cast<double>(frame_size.width)));
    const double top = std::max(0.0, std::min(box.y, static_cast<double>(frame_size.height)));
    const double right = std::max(0.0, std::min(box.x + box.width, static_cast<double>(frame_size.width)));
    const double bottom = std::max(0.0, std::min(box.y + box.height, static_cast<double>(frame_size.height)));
    return {left, top, std::max(0.0, right - left), std::max(0.0, bottom - top)};
}

cv::Mat ensure_2d_output(const cv::Mat& output) {
    if (output.dims == 2) {
        return output;
    }

    if (output.dims == 3 && output.size[0] == 1) {
        return output.reshape(1, output.size[1]);
    }

    throw std::runtime_error("Unsupported YOLO output tensor shape.");
}

std::vector<DetectionBox> decode_yolo_output(
    const cv::Mat& output,
    const LetterboxInfo& letterbox_info,
    const cv::Size& frame_size,
    float score_threshold,
    float nms_threshold,
    int top_k) {
    cv::Mat reshaped = ensure_2d_output(output);
    if (reshaped.rows < reshaped.cols) {
        reshaped = reshaped.t();
    }

    std::vector<DetectionBox> detections;
    const bool end_to_end_format = reshaped.cols == 6;
    if (end_to_end_format) {
        for (int row_index = 0; row_index < reshaped.rows; ++row_index) {
            const float* row = reshaped.ptr<float>(row_index);
            const float confidence = row[4];
            const int class_id = static_cast<int>(std::lround(row[5]));
            if (class_id != kPersonClassId || confidence < score_threshold) {
                continue;
            }

            const double raw_left = row[0];
            const double raw_top = row[1];
            const double raw_right = row[2];
            const double raw_bottom = row[3];

            const bool normalized = std::max(
                {std::abs(raw_left), std::abs(raw_top), std::abs(raw_right), std::abs(raw_bottom)}) <= 2.0;
            const double scale_x = normalized ? static_cast<double>(frame_size.width) : 1.0;
            const double scale_y = normalized ? static_cast<double>(frame_size.height) : 1.0;

            const double left = (raw_left * scale_x - letterbox_info.pad_x) / letterbox_info.ratio;
            const double top = (raw_top * scale_y - letterbox_info.pad_y) / letterbox_info.ratio;
            const double right = (raw_right * scale_x - letterbox_info.pad_x) / letterbox_info.ratio;
            const double bottom = (raw_bottom * scale_y - letterbox_info.pad_y) / letterbox_info.ratio;

            cv::Rect2d box(left, top, right - left, bottom - top);
            box = clip_box(box, frame_size);
            if (box.width <= 0.0 || box.height <= 0.0) {
                continue;
            }

            detections.push_back({
                static_cast<float>(box.x),
                static_cast<float>(box.y),
                static_cast<float>(box.width),
                static_cast<float>(box.height),
                confidence,
                class_id,
            });
        }
    } else {
        std::vector<cv::Rect2d> candidate_boxes;
        std::vector<float> candidate_scores;
        candidate_boxes.reserve(reshaped.rows);
        candidate_scores.reserve(reshaped.rows);

        for (int row_index = 0; row_index < reshaped.rows; ++row_index) {
            const float* row = reshaped.ptr<float>(row_index);
            if (reshaped.cols < 5) {
                throw std::runtime_error("YOLO output tensor does not contain enough columns.");
            }

            float best_score = 0.0f;
            int best_class_id = 0;
            for (int class_id = 0; class_id < reshaped.cols - 4; ++class_id) {
                const float score = row[4 + class_id];
                if (score > best_score) {
                    best_score = score;
                    best_class_id = class_id;
                }
            }

            if (best_class_id != kPersonClassId || best_score < score_threshold) {
                continue;
            }

            const double raw_center_x = row[0];
            const double raw_center_y = row[1];
            const double raw_width = row[2];
            const double raw_height = row[3];

            const bool normalized = std::max(
                {std::abs(raw_center_x), std::abs(raw_center_y), std::abs(raw_width), std::abs(raw_height)}) <= 2.0;
            const double scale_x = normalized ? static_cast<double>(frame_size.width) : 1.0;
            const double scale_y = normalized ? static_cast<double>(frame_size.height) : 1.0;

            const double center_x = (raw_center_x * scale_x - letterbox_info.pad_x) / letterbox_info.ratio;
            const double center_y = (raw_center_y * scale_y - letterbox_info.pad_y) / letterbox_info.ratio;
            const double box_width = (raw_width * scale_x) / letterbox_info.ratio;
            const double box_height = (raw_height * scale_y) / letterbox_info.ratio;

            const double left = center_x - box_width * 0.5;
            const double top = center_y - box_height * 0.5;
            const double right = center_x + box_width * 0.5;
            const double bottom = center_y + box_height * 0.5;

            cv::Rect2d box(left, top, right - left, bottom - top);
            box = clip_box(box, frame_size);
            if (box.width <= 0.0 || box.height <= 0.0) {
                continue;
            }

            candidate_boxes.push_back(box);
            candidate_scores.push_back(best_score);
        }

        if (!candidate_boxes.empty()) {
            std::vector<int> keep_indices;
            cv::dnn::NMSBoxes(
                candidate_boxes,
                candidate_scores,
                score_threshold,
                nms_threshold,
                keep_indices,
                1.0f,
                top_k);

            detections.reserve(keep_indices.size());
            for (const int index : keep_indices) {
                const cv::Rect2d& box = candidate_boxes[index];
                detections.push_back({
                    static_cast<float>(box.x),
                    static_cast<float>(box.y),
                    static_cast<float>(box.width),
                    static_cast<float>(box.height),
                    candidate_scores[index],
                    0,
                });
            }
        }
    }

    std::sort(
        detections.begin(),
        detections.end(),
        [](const DetectionBox& lhs, const DetectionBox& rhs) { return lhs.confidence > rhs.confidence; });

    return detections;
}

template <typename Fn>
Fn api_function(const ort::OrtApi* api, ort::OrtFunction function) {
    (void)api;
    return reinterpret_cast<Fn>(function);
}

std::string status_message(const ort::OrtApi* api, OrtStatus* status) {
    if (status == nullptr) {
        return {};
    }

    std::string message = "unknown ONNX Runtime error";
    if (api != nullptr && api->GetErrorMessage != nullptr) {
        using GetErrorMessageFn = const char* (*)(OrtStatus*);
        const char* raw_message = api_function<GetErrorMessageFn>(api, api->GetErrorMessage)(status);
        if (raw_message != nullptr) {
            message = raw_message;
        }
    }

    if (api != nullptr && api->ReleaseStatus != nullptr) {
        using ReleaseStatusFn = void (*)(OrtStatus*);
        api_function<ReleaseStatusFn>(api, api->ReleaseStatus)(status);
    }

    return message;
}

void throw_on_status(const ort::OrtApi* api, OrtStatus* status, const std::string& context) {
    if (status != nullptr) {
        throw std::runtime_error(context + ": " + status_message(api, status));
    }
}

}  // namespace

class OrtYoloSession {
    using SessionGetCountFn = OrtStatus* (*)(const OrtSession*, std::size_t*);
    using SessionGetNameFn = OrtStatus* (*)(const OrtSession*, std::size_t, OrtAllocator*, char**);

public:
    explicit OrtYoloSession(const std::string& model_path, int device_id)
        : library_(open_ort_library()) {
        using GetApiBaseFn = const ort::OrtApiBase* (*)(void);
        auto get_api_base = load_symbol<GetApiBaseFn>(library_, "OrtGetApiBase");
        const ort::OrtApiBase* api_base = get_api_base();
        if (api_base == nullptr || api_base->GetApi == nullptr) {
            throw std::runtime_error("OrtGetApiBase returned an invalid API base.");
        }

        api_ = api_base->GetApi(ort::kOrtApiVersion);
        if (api_ == nullptr) {
            throw std::runtime_error("Loaded ONNX Runtime library does not expose API version 24.");
        }

        using CreateEnvFn = OrtStatus* (*)(ort::OrtLoggingLevel, const char*, OrtEnv**);
        using CreateSessionOptionsFn = OrtStatus* (*)(OrtSessionOptions**);
        using CreateSessionFn = OrtStatus* (*)(OrtEnv*, const char*, const OrtSessionOptions*, OrtSession**);
        using CreateMemoryInfoFn = OrtStatus* (*)(const char*, ort::OrtAllocatorType, int, ort::OrtMemType, OrtMemoryInfo**);
        using CreateTensorWithDataAsOrtValueFn = OrtStatus* (*)(
            const OrtMemoryInfo*,
            void*,
            std::size_t,
            const std::int64_t*,
            std::size_t,
            ort::ONNXTensorElementDataType,
            OrtValue**);
        using RunFn = OrtStatus* (*)(
            OrtSession*,
            const OrtRunOptions*,
            const char* const*,
            const OrtValue* const*,
            std::size_t,
            const char* const*,
            std::size_t,
            OrtValue**);
        using GetTensorMutableDataFn = OrtStatus* (*)(const OrtValue*, void**);
        using GetAllocatorWithDefaultOptionsFn = OrtStatus* (*)(OrtAllocator**);
        using AllocatorFreeFn = void (*)(OrtAllocator*, void*);
        using GetTensorTypeAndShapeFn = OrtStatus* (*)(const OrtValue*, OrtTensorTypeAndShapeInfo**);
        using GetDimensionsCountFn = OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, std::size_t*);
        using GetDimensionsFn = OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, std::int64_t*, std::size_t);
        using GetTensorElementTypeFn = OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, ort::ONNXTensorElementDataType*);
        using ReleaseTensorTypeAndShapeInfoFn = void (*)(OrtTensorTypeAndShapeInfo*);
        using ReleaseEnvFn = void (*)(OrtEnv*);
        using ReleaseSessionFn = void (*)(OrtSession*);
        using ReleaseSessionOptionsFn = void (*)(OrtSessionOptions*);
        using ReleaseValueFn = void (*)(OrtValue*);
        using ReleaseMemoryInfoFn = void (*)(OrtMemoryInfo*);
        using AppendCudaFn = OrtStatus* (*)(OrtSessionOptions*, int);

        create_env_ = api_function<CreateEnvFn>(api_, api_->CreateEnv);
        create_session_options_ = api_function<CreateSessionOptionsFn>(api_, api_->CreateSessionOptions);
        create_session_ = api_function<CreateSessionFn>(api_, api_->CreateSession);
        create_memory_info_ = api_function<CreateMemoryInfoFn>(api_, api_->CreateMemoryInfo);
        create_tensor_with_data_ = api_function<CreateTensorWithDataAsOrtValueFn>(api_, api_->CreateTensorWithDataAsOrtValue);
        run_ = api_function<RunFn>(api_, api_->Run);
        get_tensor_mutable_data_ = api_function<GetTensorMutableDataFn>(api_, api_->GetTensorMutableData);
        session_get_input_count_ = api_function<SessionGetCountFn>(api_, api_->SessionGetInputCount);
        session_get_output_count_ = api_function<SessionGetCountFn>(api_, api_->SessionGetOutputCount);
        session_get_input_name_ = api_function<SessionGetNameFn>(api_, api_->SessionGetInputName);
        session_get_output_name_ = api_function<SessionGetNameFn>(api_, api_->SessionGetOutputName);
        get_allocator_with_default_options_ = api_function<GetAllocatorWithDefaultOptionsFn>(
            api_, api_->GetAllocatorWithDefaultOptions);
        allocator_free_ = api_function<AllocatorFreeFn>(api_, api_->AllocatorFree);
        get_tensor_type_and_shape_ = api_function<GetTensorTypeAndShapeFn>(api_, api_->GetTensorTypeAndShape);
        get_dimensions_count_ = api_function<GetDimensionsCountFn>(api_, api_->GetDimensionsCount);
        get_dimensions_ = api_function<GetDimensionsFn>(api_, api_->GetDimensions);
        get_tensor_element_type_ = api_function<GetTensorElementTypeFn>(api_, api_->GetTensorElementType);
        release_tensor_type_and_shape_info_ = api_function<ReleaseTensorTypeAndShapeInfoFn>(
            api_, api_->ReleaseTensorTypeAndShapeInfo);
        release_env_ = api_function<ReleaseEnvFn>(api_, api_->ReleaseEnv);
        release_session_ = api_function<ReleaseSessionFn>(api_, api_->ReleaseSession);
        release_session_options_ = api_function<ReleaseSessionOptionsFn>(api_, api_->ReleaseSessionOptions);
        release_value_ = api_function<ReleaseValueFn>(api_, api_->ReleaseValue);
        release_memory_info_ = api_function<ReleaseMemoryInfoFn>(api_, api_->ReleaseMemoryInfo);
        const auto append_cuda = load_symbol<AppendCudaFn>(library_, "OrtSessionOptionsAppendExecutionProvider_CUDA");

        OrtStatus* status = nullptr;

        OrtEnv* raw_env = nullptr;
        status = create_env_(ort::ORT_LOGGING_LEVEL_WARNING, "rescue", &raw_env);
        throw_on_status(api_, status, "Failed to create ONNX Runtime environment");
        env_ = make_api_ptr(raw_env, release_env_);

        OrtSessionOptions* raw_options = nullptr;
        status = create_session_options_(&raw_options);
        throw_on_status(api_, status, "Failed to create ONNX Runtime session options");
        session_options_ = make_api_ptr(raw_options, release_session_options_);

        preload_cuda_support_libraries();
        status = append_cuda(session_options_.get(), device_id);
        throw_on_status(api_, status, "Failed to append CUDA execution provider");

        OrtSession* raw_session = nullptr;
        status = create_session_(env_.get(), model_path.c_str(), session_options_.get(), &raw_session);
        throw_on_status(api_, status, "Failed to create ONNX Runtime session");
        session_ = make_api_ptr(raw_session, release_session_);

        OrtAllocator* raw_allocator = nullptr;
        status = get_allocator_with_default_options_(&raw_allocator);
        throw_on_status(api_, status, "Failed to acquire ORT default allocator");
        allocator_ = raw_allocator;

        input_name_ = resolve_io_name(
            "RESCUE_ORT_INPUT_NAME",
            input_name_.c_str(),
            session_get_input_count_,
            session_get_input_name_,
            "input");
        output_name_ = resolve_io_name(
            "RESCUE_ORT_OUTPUT_NAME",
            output_name_.c_str(),
            session_get_output_count_,
            session_get_output_name_,
            "output");

        OrtMemoryInfo* raw_cuda_memory_info = nullptr;
        status = create_memory_info_("Cuda", ort::OrtDeviceAllocator, device_id, ort::OrtMemTypeDefault, &raw_cuda_memory_info);
        throw_on_status(api_, status, "Failed to create CUDA memory info");
        cuda_memory_info_ = make_api_ptr(raw_cuda_memory_info, release_memory_info_);
    }

    ~OrtYoloSession() {
        session_.reset();
        session_options_.reset();
        cuda_memory_info_.reset();
        env_.reset();
        if (library_ != nullptr) {
            dlclose(library_);
            library_ = nullptr;
        }
    }

    std::vector<DetectionBox> infer(
        const GpuFrameArtifacts& frame_artifacts,
        float score_threshold,
        float nms_threshold,
        int top_k) const {
        if (frame_artifacts.device_input_tensor == nullptr || frame_artifacts.input_tensor_bytes == 0) {
            throw std::runtime_error("Empty CUDA tensor passed to person detector.");
        }

        const LetterboxInfo letterbox_info{
            frame_artifacts.resize_ratio,
            frame_artifacts.pad_x,
            frame_artifacts.pad_y,
        };
        const int64_t input_shape[] = {
            1,
            3,
            frame_artifacts.model_input_size.height,
            frame_artifacts.model_input_size.width,
        };

        OrtValue* raw_input = nullptr;
        OrtStatus* status = create_tensor_with_data_(
            cuda_memory_info_.get(),
            frame_artifacts.device_input_tensor,
            frame_artifacts.input_tensor_bytes,
            input_shape,
            4,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &raw_input);
        throw_on_status(api_, status, "Failed to create ORT input tensor");
        auto input_tensor = make_api_ptr(raw_input, release_value_);

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        const OrtValue* input_values[] = {input_tensor.get()};
        OrtValue* output_values[] = {nullptr};

        status = run_(
            session_.get(),
            nullptr,
            input_names,
            input_values,
            1,
            output_names,
            1,
            output_values);
        throw_on_status(api_, status, "Failed to run ORT inference");

        auto output_tensor = make_api_ptr(output_values[0], release_value_);

        void* output_data_raw = nullptr;
        status = get_tensor_mutable_data_(output_tensor.get(), &output_data_raw);
        throw_on_status(api_, status, "Failed to get ORT output tensor data");

        auto* output_data = static_cast<float*>(output_data_raw);
        const cv::Mat output_view = wrap_output_tensor(output_tensor.get(), output_data);
        return decode_yolo_output(
            output_view,
            letterbox_info,
            frame_artifacts.original_frame_size,
            score_threshold,
            nms_threshold,
            top_k);
    }

    std::string resolve_io_name(
        const char* env_key,
        const char* fallback,
        SessionGetCountFn get_count,
        SessionGetNameFn get_name,
        const char* label) const {
        const std::string override = read_env_string(env_key);
        if (!override.empty()) {
            return override;
        }

        if (get_count == nullptr || get_name == nullptr || allocator_ == nullptr || allocator_free_ == nullptr) {
            if (fallback != nullptr && fallback[0] != '\0') {
                return fallback;
            }
            throw std::runtime_error(std::string("Unable to resolve ORT ") + label + " name.");
        }

        std::size_t count = 0;
        OrtStatus* status = get_count(session_.get(), &count);
        throw_on_status(api_, status, std::string("Failed to query ORT ") + label + " count");
        if (count == 0) {
            throw std::runtime_error(std::string("ORT session exposes no ") + label + "s.");
        }

        if (count > 1) {
            std::clog << "Warning: ORT session exposes " << count << ' ' << label
                      << "s; using index 0. Set " << env_key << " to override." << '\n';
        }

        char* raw_name = nullptr;
        status = get_name(session_.get(), 0, allocator_, &raw_name);
        throw_on_status(api_, status, std::string("Failed to query ORT ") + label + " name");
        if (raw_name == nullptr || raw_name[0] == '\0') {
            throw std::runtime_error(std::string("ORT returned an empty ") + label + " name.");
        }

        std::string resolved = raw_name;
        allocator_free_(allocator_, raw_name);
        return resolved;
    }

    cv::Mat wrap_output_tensor(const OrtValue* output_tensor, float* output_data) const {
        OrtTensorTypeAndShapeInfo* raw_info = nullptr;
        OrtStatus* status = get_tensor_type_and_shape_(output_tensor, &raw_info);
        throw_on_status(api_, status, "Failed to read ORT output tensor shape");
        auto info = make_api_ptr(raw_info, release_tensor_type_and_shape_info_);

        ort::ONNXTensorElementDataType element_type = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        status = get_tensor_element_type_(info.get(), &element_type);
        throw_on_status(api_, status, "Failed to read ORT output tensor element type");
        if (element_type != ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ORT output tensor is not float.");
        }

        std::size_t dim_count = 0;
        status = get_dimensions_count_(info.get(), &dim_count);
        throw_on_status(api_, status, "Failed to read ORT output rank");
        if (dim_count == 0 || dim_count > 3) {
            throw std::runtime_error("ORT output rank is not supported by the YOLO decoder.");
        }

        std::vector<std::int64_t> dims(dim_count, 0);
        status = get_dimensions_(info.get(), dims.data(), dim_count);
        throw_on_status(api_, status, "Failed to read ORT output dimensions");

        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] <= 0 || dims[i] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("ORT output dimensions are invalid for decoding.");
            }
        }

        if (dim_count == 2) {
            return cv::Mat(static_cast<int>(dims[0]), static_cast<int>(dims[1]), CV_32F, output_data);
        }

        int sizes[3] = {
            static_cast<int>(dims[0]),
            static_cast<int>(dims[1]),
            static_cast<int>(dims[2]),
        };
        return cv::Mat(3, sizes, CV_32F, output_data);
    }

private:
    void* library_ = nullptr;
    const ort::OrtApi* api_ = nullptr;
    ApiPtr<OrtEnv> env_;
    ApiPtr<OrtSessionOptions> session_options_;
    ApiPtr<OrtSession> session_;
    using CreateEnvFn = OrtStatus* (*)(ort::OrtLoggingLevel, const char*, OrtEnv**);
    using CreateSessionOptionsFn = OrtStatus* (*)(OrtSessionOptions**);
    using CreateSessionFn = OrtStatus* (*)(OrtEnv*, const char*, const OrtSessionOptions*, OrtSession**);
    using CreateMemoryInfoFn = OrtStatus* (*)(const char*, ort::OrtAllocatorType, int, ort::OrtMemType, OrtMemoryInfo**);
    using CreateTensorWithDataAsOrtValueFn = OrtStatus* (*)(
        const OrtMemoryInfo*,
        void*,
        std::size_t,
        const std::int64_t*,
        std::size_t,
        ort::ONNXTensorElementDataType,
        OrtValue**);
    using RunFn = OrtStatus* (*)(
        OrtSession*,
        const OrtRunOptions*,
        const char* const*,
        const OrtValue* const*,
        std::size_t,
        const char* const*,
        std::size_t,
        OrtValue**);
    using GetTensorMutableDataFn = OrtStatus* (*)(const OrtValue*, void**);
    using GetAllocatorWithDefaultOptionsFn = OrtStatus* (*)(OrtAllocator**);
    using AllocatorFreeFn = void (*)(OrtAllocator*, void*);
    using GetTensorTypeAndShapeFn = OrtStatus* (*)(const OrtValue*, OrtTensorTypeAndShapeInfo**);
    using GetDimensionsCountFn = OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, std::size_t*);
    using GetDimensionsFn = OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, std::int64_t*, std::size_t);
    using GetTensorElementTypeFn = OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, ort::ONNXTensorElementDataType*);
    using ReleaseTensorTypeAndShapeInfoFn = void (*)(OrtTensorTypeAndShapeInfo*);
    using ReleaseEnvFn = void (*)(OrtEnv*);
    using ReleaseSessionFn = void (*)(OrtSession*);
    using ReleaseSessionOptionsFn = void (*)(OrtSessionOptions*);
    using ReleaseValueFn = void (*)(OrtValue*);
    using ReleaseMemoryInfoFn = void (*)(OrtMemoryInfo*);

    CreateEnvFn create_env_ = nullptr;
    CreateSessionOptionsFn create_session_options_ = nullptr;
    CreateSessionFn create_session_ = nullptr;
    CreateMemoryInfoFn create_memory_info_ = nullptr;
    CreateTensorWithDataAsOrtValueFn create_tensor_with_data_ = nullptr;
    RunFn run_ = nullptr;
    GetTensorMutableDataFn get_tensor_mutable_data_ = nullptr;
    SessionGetCountFn session_get_input_count_ = nullptr;
    SessionGetCountFn session_get_output_count_ = nullptr;
    SessionGetNameFn session_get_input_name_ = nullptr;
    SessionGetNameFn session_get_output_name_ = nullptr;
    GetAllocatorWithDefaultOptionsFn get_allocator_with_default_options_ = nullptr;
    AllocatorFreeFn allocator_free_ = nullptr;
    GetTensorTypeAndShapeFn get_tensor_type_and_shape_ = nullptr;
    GetDimensionsCountFn get_dimensions_count_ = nullptr;
    GetDimensionsFn get_dimensions_ = nullptr;
    GetTensorElementTypeFn get_tensor_element_type_ = nullptr;
    ReleaseTensorTypeAndShapeInfoFn release_tensor_type_and_shape_info_ = nullptr;
    ReleaseEnvFn release_env_ = nullptr;
    ReleaseSessionFn release_session_ = nullptr;
    ReleaseSessionOptionsFn release_session_options_ = nullptr;
    ReleaseValueFn release_value_ = nullptr;
    ReleaseMemoryInfoFn release_memory_info_ = nullptr;
    ApiPtr<OrtMemoryInfo> cuda_memory_info_;
    OrtAllocator* allocator_ = nullptr;
    std::string input_name_ = "images";
    std::string output_name_ = "output0";
};

class CudaPeopleDetectorImpl : public CudaPeopleDetector {
public:
    explicit CudaPeopleDetectorImpl(const AppConfig& config);
    ~CudaPeopleDetectorImpl() override;
    std::vector<DetectionBox> detect(const GpuFrameArtifacts& frame_artifacts) override;
    const std::string& backend_name() const noexcept override;
    const cv::Size& input_size() const noexcept override;

private:
    std::unique_ptr<OrtYoloSession> ort_session_;
    cv::Size input_size_{};
    float score_threshold_ = 0.2f;
    float nms_threshold_ = 0.45f;
    int top_k_ = 5000;
    std::string backend_name_ = "ONNX Runtime CUDA";
};

CudaPeopleDetectorImpl::CudaPeopleDetectorImpl(const AppConfig& config)
    : score_threshold_(static_cast<float>(config.score_threshold)),
      nms_threshold_(static_cast<float>(config.nms_threshold)),
      top_k_(config.top_k) {
    if (!std::filesystem::exists(config.model_path)) {
        throw std::runtime_error(
            "YOLO11 model not found: " + config.model_path +
            ". Place a YOLO11 ONNX model there (recommended: models/yolo11s_1088.onnx) or pass --model <path>.");
    }

    if (config.resize_width <= 0 || (config.resize_width % 32) != 0) {
        throw std::runtime_error("--resize-width must be a positive multiple of 32 for the YOLO detector path.");
    }

    input_size_ = cv::Size(config.resize_width, config.resize_width);
    const int device_id = config.gpu_device_id >= 0 ? config.gpu_device_id : 0;
    ort_session_ = std::make_unique<OrtYoloSession>(config.model_path, device_id);
    backend_name_ = "CUDA preprocess + ONNX Runtime CUDA (device " + std::to_string(device_id) + ")";
}

CudaPeopleDetectorImpl::~CudaPeopleDetectorImpl() = default;

const std::string& CudaPeopleDetectorImpl::backend_name() const noexcept {
    return backend_name_;
}

const cv::Size& CudaPeopleDetectorImpl::input_size() const noexcept {
    return input_size_;
}

std::vector<DetectionBox> CudaPeopleDetectorImpl::detect(const GpuFrameArtifacts& frame_artifacts) {
    if (ort_session_ == nullptr) {
        throw std::runtime_error("ONNX Runtime CUDA backend is not available.");
    }

    return ort_session_->infer(frame_artifacts, score_threshold_, nms_threshold_, top_k_);
}

std::unique_ptr<CudaPeopleDetector> create_cuda_detector(const AppConfig& config) {
    return std::make_unique<CudaPeopleDetectorImpl>(config);
}

}  // namespace rescue
