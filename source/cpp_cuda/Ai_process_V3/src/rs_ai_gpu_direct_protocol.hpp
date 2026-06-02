#pragma once

#include <cstddef>
#include <cstdint>

namespace rs_ai_gpu_direct {

constexpr std::uint32_t kProtocolVersion = 2;

enum class ResourceMode : std::uint32_t {
    Uniform = 1,
    RoiLayered = 2,
};

enum class TensorPriority : std::uint32_t {
    Normal = 0,
    Low = 1,
    High = 2,
    Control = 3,
};

enum class TensorDType : std::uint32_t {
    Int8 = 1,
    Float16 = 2,
    Float32 = 3,
    UInt8 = 4,
};

enum class TensorEncoding : std::uint32_t {
    RawSymmetricInt8 = 1,
    RawFloat16 = 2,
    RawFloat32 = 3,
    RawMaskU8 = 4,
};

struct TensorDesc {
    const char* name = nullptr;
    std::uint32_t ndim = 0;
    std::int32_t shape[8]{};
    TensorDType dtype = TensorDType::Int8;
    TensorEncoding encoding = TensorEncoding::RawSymmetricInt8;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_size = 0;
    float scale = 1.0f;
    std::int32_t zero_point = 0;
    const char* group = nullptr;
    const char* role = nullptr;
    TensorPriority priority = TensorPriority::Normal;
    std::uint32_t full_ndim = 0;
    std::int32_t full_shape[8]{};
};

struct FrameDesc {
    std::uint32_t protocol_version = kProtocolVersion;
    ResourceMode resource_mode = ResourceMode::Uniform;
    std::uint32_t split_layer = 0;
    std::uint32_t tensor_count = 0;
    std::uint32_t frame_id = 0;
    double pts_ms = 0.0;
    std::int32_t source_h = 0;
    std::int32_t source_w = 0;
    std::int32_t network_h = 0;
    std::int32_t network_w = 0;
    std::uint64_t payload_nbytes = 0;
    const TensorDesc* tensors = nullptr;
};

struct DevicePayload {
    const std::uint8_t* data = nullptr;
    std::uint64_t nbytes = 0;
    void* cuda_stream = nullptr;
};

struct DeviceFrameView {
    FrameDesc desc;
    DevicePayload payload;
};

}  // namespace rs_ai_gpu_direct
