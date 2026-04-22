                                                                                                                                                                                                                                                                                                                                                                                                        #pragma once

#include <cstddef>
#include <cstdint>

namespace rescue::ort {

struct OrtApi;

using OrtGetApiFn = const OrtApi* (*)(std::uint32_t version);
using OrtGetVersionStringFn = const char* (*)(void);
using OrtFunction = void (*)();

enum OrtLoggingLevel : int {
    ORT_LOGGING_LEVEL_VERBOSE = 0,
    ORT_LOGGING_LEVEL_INFO = 1,
    ORT_LOGGING_LEVEL_WARNING = 2,
    ORT_LOGGING_LEVEL_ERROR = 3,
    ORT_LOGGING_LEVEL_FATAL = 4,
};

enum OrtAllocatorType : int {
    OrtDeviceAllocator = 0,
    OrtArenaAllocator = 1,
};

enum OrtMemType : int {
    OrtMemTypeDefault = 0,
};

enum ONNXTensorElementDataType : int {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16,
};

struct OrtApiBase {
    OrtGetApiFn GetApi;
    OrtGetVersionStringFn GetVersionString;
};

#define RESCUE_ORT_API_V1_FIELDS(X) \
    X(CreateStatus) \
    X(GetErrorCode) \
    X(GetErrorMessage) \
    X(CreateEnv) \
    X(CreateEnvWithCustomLogger) \
    X(EnableTelemetryEvents) \
    X(DisableTelemetryEvents) \
    X(CreateSession) \
    X(CreateSessionFromArray) \
    X(Run) \
    X(CreateSessionOptions) \
    X(SetOptimizedModelFilePath) \
    X(CloneSessionOptions) \
    X(SetSessionExecutionMode) \
    X(EnableProfiling) \
    X(DisableProfiling) \
    X(EnableMemPattern) \
    X(DisableMemPattern) \
    X(EnableCpuMemArena) \
    X(DisableCpuMemArena) \
    X(SetSessionLogId) \
    X(SetSessionLogVerbosityLevel) \
    X(SetSessionLogSeverityLevel) \
    X(SetSessionGraphOptimizationLevel) \
    X(SetIntraOpNumThreads) \
    X(SetInterOpNumThreads) \
    X(CreateCustomOpDomain) \
    X(CustomOpDomain_Add) \
    X(AddCustomOpDomain) \
    X(RegisterCustomOpsLibrary) \
    X(SessionGetInputCount) \
    X(SessionGetOutputCount) \
    X(SessionGetOverridableInitializerCount) \
    X(SessionGetInputTypeInfo) \
    X(SessionGetOutputTypeInfo) \
    X(SessionGetOverridableInitializerTypeInfo) \
    X(SessionGetInputName) \
    X(SessionGetOutputName) \
    X(SessionGetOverridableInitializerName) \
    X(CreateRunOptions) \
    X(RunOptionsSetRunLogVerbosityLevel) \
    X(RunOptionsSetRunLogSeverityLevel) \
    X(RunOptionsSetRunTag) \
    X(RunOptionsGetRunLogVerbosityLevel) \
    X(RunOptionsGetRunLogSeverityLevel) \
    X(RunOptionsGetRunTag) \
    X(RunOptionsSetTerminate) \
    X(RunOptionsUnsetTerminate) \
    X(CreateTensorAsOrtValue) \
    X(CreateTensorWithDataAsOrtValue) \
    X(IsTensor) \
    X(GetTensorMutableData) \
    X(FillStringTensor) \
    X(GetStringTensorDataLength) \
    X(GetStringTensorContent) \
    X(CastTypeInfoToTensorInfo) \
    X(GetOnnxTypeFromTypeInfo) \
    X(CreateTensorTypeAndShapeInfo) \
    X(SetTensorElementType) \
    X(SetDimensions) \
    X(GetTensorElementType) \
    X(GetDimensionsCount) \
    X(GetDimensions) \
    X(GetSymbolicDimensions) \
    X(GetTensorShapeElementCount) \
    X(GetTensorTypeAndShape) \
    X(GetTypeInfo) \
    X(GetValueType) \
    X(CreateMemoryInfo) \
    X(CreateCpuMemoryInfo) \
    X(CompareMemoryInfo) \
    X(MemoryInfoGetName) \
    X(MemoryInfoGetId) \
    X(MemoryInfoGetMemType) \
    X(MemoryInfoGetType) \
    X(AllocatorAlloc) \
    X(AllocatorFree) \
    X(AllocatorGetInfo) \
    X(GetAllocatorWithDefaultOptions) \
    X(AddFreeDimensionOverride) \
    X(GetValue) \
    X(GetValueCount) \
    X(CreateValue) \
    X(CreateOpaqueValue) \
    X(GetOpaqueValue) \
    X(KernelInfoGetAttribute_float) \
    X(KernelInfoGetAttribute_int64) \
    X(KernelInfoGetAttribute_string) \
    X(KernelContext_GetInputCount) \
    X(KernelContext_GetOutputCount) \
    X(KernelContext_GetInput) \
    X(KernelContext_GetOutput) \
    X(ReleaseEnv) \
    X(ReleaseStatus) \
    X(ReleaseMemoryInfo) \
    X(ReleaseSession) \
    X(ReleaseValue) \
    X(ReleaseRunOptions) \
    X(ReleaseTypeInfo) \
    X(ReleaseTensorTypeAndShapeInfo) \
    X(ReleaseSessionOptions) \
    X(ReleaseCustomOpDomain)

struct OrtApi {
#define RESCUE_DEFINE_API_FIELD(name) OrtFunction name;
    RESCUE_ORT_API_V1_FIELDS(RESCUE_DEFINE_API_FIELD)
#undef RESCUE_DEFINE_API_FIELD
};

inline constexpr std::uint32_t kOrtApiVersion = 24;

}  // namespace rescue::ort

extern "C" {

struct OrtStatus;
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtAllocator;
struct OrtValue;
struct OrtTensorTypeAndShapeInfo;
struct OrtRunOptions;

const rescue::ort::OrtApiBase* OrtGetApiBase(void) noexcept;

OrtStatus* OrtSessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions* options, int device_id);

}