#include "rs_ai_gpu_direct_api.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvjpeg.h>

#include "NvDecoder/NvDecoder.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/Logger.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

simplelogger::Logger* logger = nullptr;

namespace {

struct CompressorHandle {
    RsAiCompressionCodec codec = RS_AI_COMPRESSION_NONE;
    int32_t cuda_device = 0;
    size_t chunk_bytes = 1u << 16;
    nvjpegHandle_t nvjpeg = nullptr;
    nvjpegEncoderState_t encoder_state = nullptr;
    nvjpegEncoderParams_t encoder_params = nullptr;
    nvjpegJpegState_t jpeg_state = nullptr;
};

struct VideoCodecHandle {
    RsAiVideoCodec codec = RS_AI_VIDEO_CODEC_NONE;
    int32_t cuda_device = 0;
    CUdevice cu_device = 0;
    CUcontext cu_context = nullptr;
    CUstream cu_stream = nullptr;
    bool owns_context = false;
    std::unique_ptr<NvDecoder> decoder;
    std::unique_ptr<NvEncoderCuda> encoder;
    int32_t enc_width = 0;
    int32_t enc_height = 0;
    NV_ENC_BUFFER_FORMAT enc_format = NV_ENC_BUFFER_FORMAT_NV12;
};

struct AiHandle {
    std::string model_path;
    RsAiResourceMode resource_mode = RS_AI_RESOURCE_UNIFORM;
    int32_t split_layer = 0;
    int32_t image_size = 0;
    int32_t cuda_device = 0;
    std::vector<RsAiTensorDesc> tensor_descs;
    std::vector<RsAiDetection> detections;
};

struct UiRendererHandle {
    int32_t cuda_device = 0;
};

RsAiStatus cuda_status(cudaError_t status)
{
    return status == cudaSuccess ? RS_AI_STATUS_OK : RS_AI_STATUS_RUNTIME_ERROR;
}

RsAiStatus cu_status(CUresult status)
{
    return status == CUDA_SUCCESS ? RS_AI_STATUS_OK : RS_AI_STATUS_RUNTIME_ERROR;
}

RsAiStatus nvjpeg_status(nvjpegStatus_t status)
{
    switch (status) {
    case NVJPEG_STATUS_SUCCESS:
        return RS_AI_STATUS_OK;
    case NVJPEG_STATUS_INVALID_PARAMETER:
    case NVJPEG_STATUS_BAD_JPEG:
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return RS_AI_STATUS_INVALID_ARGUMENT;
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return RS_AI_STATUS_UNSUPPORTED;
    default:
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

void clear_slab(RsAiGpuSlab* slab)
{
    if (slab == nullptr) {
        return;
    }
    slab->memory.ptr = nullptr;
    slab->memory.bytes = 0;
    slab->memory.kind = RS_AI_BUFFER_UNKNOWN;
    slab->capacity_bytes = 0;
    slab->used_bytes = 0;
    slab->alignment_bytes = 0;
    slab->slab_id = 0;
    slab->cuda_stream = nullptr;
    slab->ready_event = nullptr;
}

uint64_t align_up(uint64_t value, uint32_t alignment)
{
    const uint64_t a = alignment == 0 ? 256ull : static_cast<uint64_t>(alignment);
    return (value + a - 1ull) / a * a;
}

RsAiStatus reserve_slab_bytes(RsAiGpuSlab* slab, uint64_t bytes, void** out_ptr, uint64_t* out_offset)
{
    if (slab == nullptr || slab->memory.ptr == nullptr || out_ptr == nullptr || out_offset == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t offset = align_up(slab->used_bytes, slab->alignment_bytes);
    if (bytes > slab->capacity_bytes || offset > slab->capacity_bytes - bytes) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    *out_ptr = static_cast<unsigned char*>(slab->memory.ptr) + offset;
    *out_offset = offset;
    slab->used_bytes = offset + bytes;
    return RS_AI_STATUS_OK;
}

cudaStream_t as_stream(void* stream)
{
    return static_cast<cudaStream_t>(stream);
}

int image_channels(RsAiImageFormat format)
{
    switch (format) {
    case RS_AI_IMAGE_FORMAT_BGR8:
    case RS_AI_IMAGE_FORMAT_RGB8:
        return 3;
    case RS_AI_IMAGE_FORMAT_RGBA8:
        return 4;
    default:
        return 0;
    }
}

RsAiStatus record_ready_event(RsAiGpuSlab* slab, void* stream, void** out_event)
{
    if (out_event != nullptr) {
        *out_event = nullptr;
    }
    cudaEvent_t event = nullptr;
    cudaError_t st = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    st = cudaEventRecord(event, as_stream(stream));
    if (st != cudaSuccess) {
        cudaEventDestroy(event);
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    if (out_event != nullptr) {
        *out_event = event;
    }
    if (slab != nullptr) {
        slab->ready_event = event;
    }
    return RS_AI_STATUS_OK;
}

bool nvjpeg_input_format(RsAiImageFormat fmt, nvjpegInputFormat_t* out_format, int* out_channels)
{
    if (out_format == nullptr || out_channels == nullptr) {
        return false;
    }
    switch (fmt) {
    case RS_AI_IMAGE_FORMAT_BGR8:
        *out_format = NVJPEG_INPUT_BGRI;
        *out_channels = 3;
        return true;
    case RS_AI_IMAGE_FORMAT_RGB8:
        *out_format = NVJPEG_INPUT_RGBI;
        *out_channels = 3;
        return true;
    default:
        return false;
    }
}

bool nvjpeg_output_format(RsAiImageFormat fmt, nvjpegOutputFormat_t* out_format, int* out_channels)
{
    if (out_format == nullptr || out_channels == nullptr) {
        return false;
    }
    switch (fmt) {
    case RS_AI_IMAGE_FORMAT_BGR8:
        *out_format = NVJPEG_OUTPUT_BGRI;
        *out_channels = 3;
        return true;
    case RS_AI_IMAGE_FORMAT_RGB8:
        *out_format = NVJPEG_OUTPUT_RGBI;
        *out_channels = 3;
        return true;
    default:
        return false;
    }
}

__global__ void image_to_rgba_kernel(const uint8_t* src,
                                     int width,
                                     int height,
                                     int src_pitch,
                                     int src_channels,
                                     int src_format,
                                     uint8_t* dst,
                                     int dst_pitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const uint8_t* p = src + y * src_pitch + x * src_channels;
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
    if (src_format == RS_AI_IMAGE_FORMAT_BGR8) {
        b = p[0];
        g = p[1];
        r = p[2];
    } else if (src_format == RS_AI_IMAGE_FORMAT_RGB8) {
        r = p[0];
        g = p[1];
        b = p[2];
    } else {
        r = p[0];
        g = p[1];
        b = p[2];
        a = p[3];
    }
    uint8_t* q = dst + y * dst_pitch + x * 4;
    q[0] = r;
    q[1] = g;
    q[2] = b;
    q[3] = a;
}

__global__ void draw_detections_kernel(uint8_t* rgba,
                                       int width,
                                       int height,
                                       int pitch,
                                       const RsAiDetection* detections,
                                       uint32_t count)
{
    const uint32_t det_idx = blockIdx.z;
    if (det_idx >= count) {
        return;
    }
    const RsAiDetection det = detections[det_idx];
    const int x1 = max(0, min(width - 1, static_cast<int>(det.x1 + 0.5f)));
    const int y1 = max(0, min(height - 1, static_cast<int>(det.y1 + 0.5f)));
    const int x2 = max(0, min(width - 1, static_cast<int>(det.x2 + 0.5f)));
    const int y2 = max(0, min(height - 1, static_cast<int>(det.y2 + 0.5f)));
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < x1 || x > x2 || y < y1 || y > y2) {
        return;
    }
    const int border = 2;
    const bool on_border = (x - x1 < border) || (x2 - x < border) || (y - y1 < border) || (y2 - y < border);
    if (!on_border) {
        return;
    }
    uint8_t* q = rgba + y * pitch + x * 4;
    q[0] = 0;
    q[1] = 255;
    q[2] = 64;
    q[3] = 255;
}

cudaVideoCodec nvdec_codec(RsAiVideoCodec codec)
{
    switch (codec) {
    case RS_AI_VIDEO_CODEC_H264:
        return cudaVideoCodec_H264;
    case RS_AI_VIDEO_CODEC_HEVC:
        return cudaVideoCodec_HEVC;
    case RS_AI_VIDEO_CODEC_AV1:
        return cudaVideoCodec_AV1;
    default:
        return cudaVideoCodec_NumCodecs;
    }
}

GUID nvenc_codec(RsAiVideoCodec codec)
{
    switch (codec) {
    case RS_AI_VIDEO_CODEC_H264:
        return NV_ENC_CODEC_H264_GUID;
    case RS_AI_VIDEO_CODEC_HEVC:
        return NV_ENC_CODEC_HEVC_GUID;
    case RS_AI_VIDEO_CODEC_AV1:
        return NV_ENC_CODEC_AV1_GUID;
    default:
        return GUID{};
    }
}

bool is_zero_guid(const GUID& g)
{
    GUID z{};
    return std::memcmp(&g, &z, sizeof(GUID)) == 0;
}

RsAiStatus init_driver_context(VideoCodecHandle* h, int32_t cuda_device)
{
    if (h == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    CUresult st = cuInit(0);
    if (st != CUDA_SUCCESS) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    st = cuDeviceGet(&h->cu_device, cuda_device);
    if (st != CUDA_SUCCESS) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    st = cuDevicePrimaryCtxRetain(&h->cu_context, h->cu_device);
    if (st != CUDA_SUCCESS) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    h->owns_context = true;
    h->cuda_device = cuda_device;
    return RS_AI_STATUS_OK;
}

RsAiStatus ensure_encoder(VideoCodecHandle* h, const RsAiDeviceImage* image)
{
    if (h == nullptr || image == nullptr || image->format != RS_AI_IMAGE_FORMAT_NV12 ||
        image->width <= 0 || image->height <= 0) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    if (h->encoder && h->enc_width == image->width && h->enc_height == image->height) {
        return RS_AI_STATUS_OK;
    }
    GUID codec_guid = nvenc_codec(h->codec);
    if (is_zero_guid(codec_guid)) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    try {
        h->encoder.reset(new NvEncoderCuda(
            h->cu_context,
            static_cast<uint32_t>(image->width),
            static_cast<uint32_t>(image->height),
            NV_ENC_BUFFER_FORMAT_NV12));
        NV_ENC_INITIALIZE_PARAMS init_params = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encode_config = { NV_ENC_CONFIG_VER };
        init_params.encodeConfig = &encode_config;
        h->encoder->CreateDefaultEncoderParams(
            &init_params,
            codec_guid,
            NV_ENC_PRESET_P3_GUID,
            NV_ENC_TUNING_INFO_LOW_LATENCY);
        init_params.frameRateNum = 30;
        init_params.frameRateDen = 1;
        init_params.enablePTD = 1;
        init_params.encodeConfig->gopLength = 30;
        init_params.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        init_params.encodeConfig->rcParams.averageBitRate = 8'000'000;
        h->encoder->CreateEncoder(&init_params);
        h->enc_width = image->width;
        h->enc_height = image->height;
        h->enc_format = NV_ENC_BUFFER_FORMAT_NV12;
    } catch (...) {
        h->encoder.reset();
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    return RS_AI_STATUS_OK;
}

}  // namespace

extern "C" {

#ifndef HAVE_RS_AI_LIBTORCH
RsAiStatus rs_ai_create(const char* model_path,
                        RsAiResourceMode resource_mode,
                        int32_t split_layer,
                        int32_t image_size,
                        int32_t cuda_device,
                        RsAiHandle* out_handle)
{
    if (out_handle == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    cudaError_t st = cudaSetDevice(cuda_device);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    AiHandle* h = new AiHandle();
    h->model_path = model_path != nullptr ? model_path : "";
    h->resource_mode = resource_mode;
    h->split_layer = split_layer;
    h->image_size = image_size;
    h->cuda_device = cuda_device;
    h->tensor_descs.resize(1);
    *out_handle = h;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_destroy(RsAiHandle handle)
{
    delete static_cast<AiHandle*>(handle);
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_extract_from_bgr_device(RsAiHandle handle,
                                         const uint8_t* device_bgr,
                                         int32_t height,
                                         int32_t width,
                                         int32_t stride_bytes,
                                         uint32_t frame_id,
                                         double pts_ms,
                                         void* cuda_stream,
                                         RsAiDeviceFrame* out_frame)
{
    if (handle == nullptr || device_bgr == nullptr || height <= 0 || width <= 0 || out_frame == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    AiHandle* h = static_cast<AiHandle*>(handle);
    cudaError_t st = cudaSetDevice(h->cuda_device);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(width) * 3ull;
    const uint64_t bytes = row_bytes * static_cast<uint64_t>(height);
    uint8_t* out_ptr = nullptr;
    st = cudaMalloc(&out_ptr, static_cast<size_t>(bytes));
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    cudaStream_t stream = as_stream(cuda_stream);
    st = cudaMemcpy2DAsync(out_ptr, static_cast<size_t>(row_bytes),
                           device_bgr, static_cast<size_t>(stride_bytes > 0 ? stride_bytes : row_bytes),
                           static_cast<size_t>(row_bytes), static_cast<size_t>(height),
                           cudaMemcpyDeviceToDevice, stream);
    if (st != cudaSuccess) {
        cudaFree(out_ptr);
        return RS_AI_STATUS_RUNTIME_ERROR;
    }

    h->tensor_descs.resize(1);
    RsAiTensorDesc& tensor = h->tensor_descs[0];
    tensor = {};
    tensor.name = "visual_bgr";
    tensor.group = "preview";
    tensor.role = "ai_c_abi_mock_payload";
    tensor.ndim = 3;
    tensor.shape[0] = height;
    tensor.shape[1] = width;
    tensor.shape[2] = 3;
    tensor.full_ndim = 3;
    tensor.full_shape[0] = height;
    tensor.full_shape[1] = width;
    tensor.full_shape[2] = 3;
    tensor.dtype = RS_AI_DTYPE_UINT8;
    tensor.encoding = RS_AI_ENCODING_RAW_MASK_U8;
    tensor.priority = RS_AI_PRIORITY_NORMAL;
    tensor.byte_offset = 0;
    tensor.byte_size = bytes;
    tensor.scale = 1.0f;
    tensor.zero_point = 0;

    *out_frame = {};
    out_frame->desc.protocol_version = RS_AI_GPU_DIRECT_PROTOCOL_VERSION;
    out_frame->desc.resource_mode = h->resource_mode;
    out_frame->desc.split_layer = static_cast<uint32_t>(h->split_layer);
    out_frame->desc.tensor_count = 1;
    out_frame->desc.frame_id = frame_id;
    out_frame->desc.pts_ms = pts_ms;
    out_frame->desc.source_h = height;
    out_frame->desc.source_w = width;
    out_frame->desc.network_h = h->image_size > 0 ? h->image_size : height;
    out_frame->desc.network_w = h->image_size > 0 ? h->image_size : width;
    out_frame->desc.payload_nbytes = bytes;
    out_frame->desc.tensors = h->tensor_descs.data();
    out_frame->device_payload = out_ptr;
    out_frame->device_payload_nbytes = bytes;
    out_frame->cuda_stream = stream;
    out_frame->slab_id = 0xA1C0DE01u;
    return record_ready_event(nullptr, stream, &out_frame->ready_event);
}

RsAiStatus rs_ai_detect_from_device_frame(RsAiHandle handle,
                                          const RsAiDeviceFrame* frame,
                                          float conf_threshold,
                                          float,
                                          RsAiDetectionList* out_detections)
{
    if (handle == nullptr || frame == nullptr || out_detections == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    AiHandle* h = static_cast<AiHandle*>(handle);
    h->detections.clear();
    RsAiDetection det{};
    det.frame_id = frame->desc.frame_id;
    det.class_id = 0;
    det.confidence = conf_threshold > 0.0f ? std::max(conf_threshold, 0.51f) : 0.90f;
    const float w = static_cast<float>(std::max(1, frame->desc.source_w));
    const float ht = static_cast<float>(std::max(1, frame->desc.source_h));
    det.x1 = 0.25f * w;
    det.y1 = 0.25f * ht;
    det.x2 = 0.75f * w;
    det.y2 = 0.75f * ht;
    h->detections.push_back(det);
    out_detections->count = static_cast<uint32_t>(h->detections.size());
    out_detections->detections = h->detections.data();
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_release_device_frame(RsAiHandle, RsAiDeviceFrame* frame)
{
    if (frame == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    if (frame->ready_event != nullptr && frame->slab_id == 0xA1C0DE01u) {
        cudaEventDestroy(static_cast<cudaEvent_t>(frame->ready_event));
    }
    if (frame->device_payload != nullptr && frame->slab_id == 0xA1C0DE01u) {
        cudaFree(const_cast<uint8_t*>(frame->device_payload));
    }
    *frame = {};
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_release_detections(RsAiHandle, RsAiDetectionList* detections)
{
    if (detections == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *detections = {};
    return RS_AI_STATUS_OK;
}
#endif

RsAiStatus rs_ai_create_ui_renderer(int32_t cuda_device, RsAiUiHandle* out_handle)
{
    if (out_handle == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    cudaError_t st = cudaSetDevice(cuda_device);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    UiRendererHandle* h = new UiRendererHandle();
    h->cuda_device = cuda_device;
    *out_handle = h;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_destroy_ui_renderer(RsAiUiHandle handle)
{
    delete static_cast<UiRendererHandle*>(handle);
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_allocate_slab(uint64_t capacity_bytes,
                               uint32_t alignment_bytes,
                               int32_t cuda_device,
                               void* cuda_stream,
                               RsAiGpuSlab* out_slab)
{
    if (capacity_bytes == 0 || out_slab == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    clear_slab(out_slab);
    cudaError_t st = cudaSetDevice(cuda_device);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    void* ptr = nullptr;
    st = cudaMalloc(&ptr, static_cast<size_t>(capacity_bytes));
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    out_slab->memory.ptr = ptr;
    out_slab->memory.bytes = capacity_bytes;
    out_slab->memory.kind = RS_AI_BUFFER_CUDA_DEVICE;
    out_slab->capacity_bytes = capacity_bytes;
    out_slab->used_bytes = 0;
    out_slab->alignment_bytes = alignment_bytes == 0 ? 256u : alignment_bytes;
    out_slab->slab_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) & 0xffffffffu);
    out_slab->cuda_stream = cuda_stream;
    out_slab->ready_event = nullptr;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_wrap_external_slab(void* device_ptr,
                                    uint64_t capacity_bytes,
                                    uint32_t alignment_bytes,
                                    uint32_t slab_id,
                                    void* cuda_stream,
                                    RsAiGpuSlab* out_slab)
{
    if (device_ptr == nullptr || capacity_bytes == 0 || out_slab == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    out_slab->memory.ptr = device_ptr;
    out_slab->memory.bytes = capacity_bytes;
    out_slab->memory.kind = RS_AI_BUFFER_EXTERNAL_DEVICE;
    out_slab->capacity_bytes = capacity_bytes;
    out_slab->used_bytes = 0;
    out_slab->alignment_bytes = alignment_bytes == 0 ? 256u : alignment_bytes;
    out_slab->slab_id = slab_id;
    out_slab->cuda_stream = cuda_stream;
    out_slab->ready_event = nullptr;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_release_slab(RsAiGpuSlab* slab)
{
    if (slab == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    if (slab->ready_event != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(slab->ready_event));
        slab->ready_event = nullptr;
    }
    if (slab->memory.ptr != nullptr && slab->memory.kind == RS_AI_BUFFER_CUDA_DEVICE) {
        const cudaError_t st = cudaFree(slab->memory.ptr);
        clear_slab(slab);
        return cuda_status(st);
    }
    clear_slab(slab);
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_create_video_decoder(RsAiVideoCodec codec, int32_t cuda_device, RsAiVideoCodecHandle* out_handle)
{
    if (out_handle == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    if (nvdec_codec(codec) == cudaVideoCodec_NumCodecs) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    VideoCodecHandle* h = new VideoCodecHandle();
    h->codec = codec;
    RsAiStatus status = init_driver_context(h, cuda_device);
    if (status != RS_AI_STATUS_OK) {
        delete h;
        return status;
    }
    try {
        h->decoder.reset(new NvDecoder(
            h->cu_context,
            true,
            nvdec_codec(codec),
            false,
            true,
            nullptr,
            nullptr,
            false,
            0,
            0,
            1000,
            false,
            0,
            h->cu_stream));
    } catch (...) {
        if (h->owns_context) {
            cuDevicePrimaryCtxRelease(h->cu_device);
        }
        delete h;
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    *out_handle = h;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_nvdec_decode_to_device(RsAiVideoCodecHandle handle,
                                        const uint8_t* host_bitstream,
                                        uint64_t host_bitstream_bytes,
                                        RsAiGpuSlab* output_slab,
                                        void* cuda_stream,
                                        RsAiDeviceImage* out_image)
{
    if (handle == nullptr || host_bitstream == nullptr || host_bitstream_bytes == 0 ||
        output_slab == nullptr || out_image == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    VideoCodecHandle* h = static_cast<VideoCodecHandle*>(handle);
    if (!h->decoder) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    if (host_bitstream_bytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }

    try {
        const int frames = h->decoder->Decode(host_bitstream, static_cast<int>(host_bitstream_bytes));
        if (frames <= 0) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
        uint8_t* frame = h->decoder->GetFrame();
        if (frame == nullptr) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }

        const int width = h->decoder->GetWidth();
        const int height = h->decoder->GetHeight();
        const int pitch = h->decoder->GetDeviceFramePitch();
        const int total_rows = height + h->decoder->GetChromaHeight() * h->decoder->GetNumChromaPlanes();
        const uint64_t frame_bytes = static_cast<uint64_t>(pitch) *
            static_cast<uint64_t>(total_rows) *
            static_cast<uint64_t>(h->decoder->GetBPP());

        void* out_ptr = nullptr;
        uint64_t out_offset = 0;
        RsAiStatus status = reserve_slab_bytes(output_slab, frame_bytes, &out_ptr, &out_offset);
        if (status != RS_AI_STATUS_OK) {
            return status;
        }
        (void)out_offset;

        cudaStream_t stream = as_stream(cuda_stream != nullptr ? cuda_stream : output_slab->cuda_stream);
        cudaError_t cst = cudaMemcpyAsync(out_ptr, frame, static_cast<size_t>(frame_bytes), cudaMemcpyDeviceToDevice, stream);
        if (cst != cudaSuccess) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }

        std::memset(out_image, 0, sizeof(*out_image));
        out_image->planes[0].ptr = out_ptr;
        out_image->planes[0].bytes = frame_bytes;
        out_image->planes[0].kind = RS_AI_BUFFER_CUDA_DEVICE;
        out_image->plane_count = 1;
        out_image->width = width;
        out_image->height = height;
        out_image->pitch_bytes[0] = pitch;
        out_image->format = RS_AI_IMAGE_FORMAT_NV12;
        out_image->cuda_stream = stream;
        return record_ready_event(output_slab, stream, &out_image->ready_event);
    } catch (...) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

RsAiStatus rs_ai_create_video_encoder(RsAiVideoCodec codec, int32_t cuda_device, RsAiVideoCodecHandle* out_handle)
{
    if (out_handle == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    if (is_zero_guid(nvenc_codec(codec))) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    VideoCodecHandle* h = new VideoCodecHandle();
    h->codec = codec;
    RsAiStatus status = init_driver_context(h, cuda_device);
    if (status != RS_AI_STATUS_OK) {
        delete h;
        return status;
    }
    *out_handle = h;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_nvenc_encode_from_device(RsAiVideoCodecHandle handle,
                                          const RsAiDeviceImage* image,
                                          uint8_t* host_bitstream,
                                          uint64_t host_capacity_bytes,
                                          uint64_t* out_bytes,
                                          void* cuda_stream)
{
    if (handle == nullptr || image == nullptr || host_bitstream == nullptr || out_bytes == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_bytes = 0;
    if (image->format != RS_AI_IMAGE_FORMAT_NV12 || image->plane_count < 1 || image->planes[0].ptr == nullptr) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    VideoCodecHandle* h = static_cast<VideoCodecHandle*>(handle);
    RsAiStatus status = ensure_encoder(h, image);
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    try {
        cudaStream_t cuda_st = as_stream(cuda_stream != nullptr ? cuda_stream : image->cuda_stream);
        if (cuda_st != nullptr) {
            h->encoder->SetIOCudaStreams(reinterpret_cast<NV_ENC_CUSTREAM_PTR>(cuda_st),
                                         reinterpret_cast<NV_ENC_CUSTREAM_PTR>(cuda_st));
        }
        const NvEncInputFrame* input = h->encoder->GetNextInputFrame();
        NvEncoderCuda::CopyToDeviceFrame(
            h->cu_context,
            image->planes[0].ptr,
            static_cast<uint32_t>(image->pitch_bytes[0] > 0 ? image->pitch_bytes[0] : image->width),
            reinterpret_cast<CUdeviceptr>(input->inputPtr),
            input->pitch,
            h->encoder->GetEncodeWidth(),
            h->encoder->GetEncodeHeight(),
            CU_MEMORYTYPE_DEVICE,
            input->bufferFormat,
            input->chromaOffsets,
            input->numChromaPlanes,
            false,
            reinterpret_cast<CUstream>(cuda_st));

        std::vector<NvEncOutputFrame> packets;
        h->encoder->EncodeFrame(packets);

        uint64_t total = 0;
        for (const NvEncOutputFrame& packet : packets) {
            total += static_cast<uint64_t>(packet.frame.size());
        }
        if (total > host_capacity_bytes) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
        uint64_t offset = 0;
        for (const NvEncOutputFrame& packet : packets) {
            if (!packet.frame.empty()) {
                std::memcpy(host_bitstream + offset, packet.frame.data(), packet.frame.size());
                offset += static_cast<uint64_t>(packet.frame.size());
            }
        }
        *out_bytes = total;
        return RS_AI_STATUS_OK;
    } catch (...) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

RsAiStatus rs_ai_destroy_video_codec(RsAiVideoCodecHandle handle)
{
    if (handle == nullptr) {
        return RS_AI_STATUS_OK;
    }
    VideoCodecHandle* h = static_cast<VideoCodecHandle*>(handle);
    try {
        if (h->encoder) {
            std::vector<NvEncOutputFrame> flushed;
            h->encoder->EndEncode(flushed);
            h->encoder->DestroyEncoder();
        }
    } catch (...) {
    }
    h->encoder.reset();
    h->decoder.reset();
    if (h->owns_context) {
        cuDevicePrimaryCtxRelease(h->cu_device);
        h->owns_context = false;
    }
    delete h;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_create_compressor(RsAiCompressionCodec codec, int32_t cuda_device, RsAiCompressionHandle* out_handle)
{
    if (out_handle == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    if (codec != RS_AI_COMPRESSION_NVJPEG &&
        codec != RS_AI_COMPRESSION_NVCOMP_LZ4 &&
        codec != RS_AI_COMPRESSION_NVCOMP_GDEFLATE) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    cudaError_t cst = cudaSetDevice(cuda_device);
    if (cst != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    CompressorHandle* h = new CompressorHandle();
    h->codec = codec;
    h->cuda_device = cuda_device;
    if (codec != RS_AI_COMPRESSION_NVJPEG) {
        *out_handle = h;
        return RS_AI_STATUS_OK;
    }

    RsAiStatus status = nvjpeg_status(nvjpegCreateSimple(&h->nvjpeg));
    if (status != RS_AI_STATUS_OK) {
        delete h;
        return status;
    }
    status = nvjpeg_status(nvjpegEncoderStateCreate(h->nvjpeg, &h->encoder_state, nullptr));
    if (status != RS_AI_STATUS_OK) {
        nvjpegDestroy(h->nvjpeg);
        delete h;
        return status;
    }
    status = nvjpeg_status(nvjpegEncoderParamsCreate(h->nvjpeg, &h->encoder_params, nullptr));
    if (status != RS_AI_STATUS_OK) {
        nvjpegEncoderStateDestroy(h->encoder_state);
        nvjpegDestroy(h->nvjpeg);
        delete h;
        return status;
    }
    status = nvjpeg_status(nvjpegJpegStateCreate(h->nvjpeg, &h->jpeg_state));
    if (status != RS_AI_STATUS_OK) {
        nvjpegEncoderParamsDestroy(h->encoder_params);
        nvjpegEncoderStateDestroy(h->encoder_state);
        nvjpegDestroy(h->nvjpeg);
        delete h;
        return status;
    }
    *out_handle = h;
    return RS_AI_STATUS_OK;
}

RsAiStatus rs_ai_nvjpeg_encode_image_to_slab(RsAiCompressionHandle handle,
                                             const RsAiDeviceImage* image,
                                             int32_t quality,
                                             RsAiGpuSlab* output_slab,
                                             void* cuda_stream,
                                             RsAiDeviceSpan* out_jpeg)
{
    if (handle == nullptr || image == nullptr || output_slab == nullptr || out_jpeg == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    CompressorHandle* h = static_cast<CompressorHandle*>(handle);
    if (h->codec != RS_AI_COMPRESSION_NVJPEG || h->nvjpeg == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    if (image->plane_count < 1 || image->planes[0].ptr == nullptr || image->width <= 0 || image->height <= 0) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }

    nvjpegInputFormat_t input_format{};
    int channels = 0;
    if (!nvjpeg_input_format(image->format, &input_format, &channels)) {
        return RS_AI_STATUS_UNSUPPORTED;
    }

    cudaStream_t stream = as_stream(cuda_stream != nullptr ? cuda_stream : image->cuda_stream);
    RsAiStatus status = nvjpeg_status(nvjpegEncoderParamsSetQuality(
        h->encoder_params,
        quality <= 0 ? 85 : quality,
        stream));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }
    status = nvjpeg_status(nvjpegEncoderParamsSetEncoding(
        h->encoder_params,
        NVJPEG_ENCODING_BASELINE_DCT,
        stream));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }
    status = nvjpeg_status(nvjpegEncoderParamsSetSamplingFactors(
        h->encoder_params,
        NVJPEG_CSS_420,
        stream));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    nvjpegImage_t src{};
    src.channel[0] = static_cast<unsigned char*>(image->planes[0].ptr);
    src.pitch[0] = image->pitch_bytes[0] > 0
        ? static_cast<size_t>(image->pitch_bytes[0])
        : static_cast<size_t>(image->width * channels);

    status = nvjpeg_status(nvjpegEncodeImage(
        h->nvjpeg,
        h->encoder_state,
        h->encoder_params,
        &src,
        input_format,
        image->width,
        image->height,
        stream));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    size_t max_len = 0;
    status = nvjpeg_status(nvjpegEncodeGetBufferSize(
        h->nvjpeg,
        h->encoder_params,
        image->width,
        image->height,
        &max_len));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    void* out_ptr = nullptr;
    uint64_t out_offset = 0;
    status = reserve_slab_bytes(output_slab, static_cast<uint64_t>(max_len), &out_ptr, &out_offset);
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    size_t actual_len = max_len;
    status = nvjpeg_status(nvjpegEncodeRetrieveBitstreamDevice(
        h->nvjpeg,
        h->encoder_state,
        static_cast<unsigned char*>(out_ptr),
        &actual_len,
        stream));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }
    output_slab->used_bytes = out_offset + static_cast<uint64_t>(actual_len);
    output_slab->cuda_stream = stream;
    out_jpeg->ptr = out_ptr;
    out_jpeg->bytes = static_cast<uint64_t>(actual_len);
    out_jpeg->kind = RS_AI_BUFFER_CUDA_DEVICE;
    return record_ready_event(output_slab, stream, nullptr);
}

RsAiStatus rs_ai_nvjpeg_decode_host_to_slab(RsAiCompressionHandle handle,
                                            const uint8_t* host_jpeg,
                                            uint64_t host_jpeg_bytes,
                                            RsAiImageFormat output_format,
                                            RsAiGpuSlab* output_slab,
                                            void* cuda_stream,
                                            RsAiDeviceImage* out_image)
{
    if (handle == nullptr || host_jpeg == nullptr || host_jpeg_bytes == 0 || output_slab == nullptr || out_image == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    CompressorHandle* h = static_cast<CompressorHandle*>(handle);
    if (h->codec != RS_AI_COMPRESSION_NVJPEG || h->nvjpeg == nullptr || h->jpeg_state == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }

    nvjpegOutputFormat_t nv_output{};
    int out_channels = 0;
    if (!nvjpeg_output_format(output_format, &nv_output, &out_channels)) {
        return RS_AI_STATUS_UNSUPPORTED;
    }

    int components = 0;
    nvjpegChromaSubsampling_t subsampling{};
    int widths[NVJPEG_MAX_COMPONENT]{};
    int heights[NVJPEG_MAX_COMPONENT]{};
    RsAiStatus status = nvjpeg_status(nvjpegGetImageInfo(
        h->nvjpeg,
        host_jpeg,
        static_cast<size_t>(host_jpeg_bytes),
        &components,
        &subsampling,
        widths,
        heights));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }
    if (widths[0] <= 0 || heights[0] <= 0) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }

    const int width = widths[0];
    const int height = heights[0];
    const uint64_t pitch = static_cast<uint64_t>(width * out_channels);
    const uint64_t bytes = pitch * static_cast<uint64_t>(height);
    void* out_ptr = nullptr;
    uint64_t out_offset = 0;
    status = reserve_slab_bytes(output_slab, bytes, &out_ptr, &out_offset);
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    cudaStream_t stream = as_stream(cuda_stream);
    nvjpegImage_t dst{};
    dst.channel[0] = static_cast<unsigned char*>(out_ptr);
    dst.pitch[0] = static_cast<size_t>(pitch);
    status = nvjpeg_status(nvjpegDecode(
        h->nvjpeg,
        h->jpeg_state,
        host_jpeg,
        static_cast<size_t>(host_jpeg_bytes),
        nv_output,
        &dst,
        stream));
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    *out_image = {};
    out_image->planes[0].ptr = out_ptr;
    out_image->planes[0].bytes = bytes;
    out_image->planes[0].kind = RS_AI_BUFFER_CUDA_DEVICE;
    out_image->plane_count = 1;
    out_image->width = width;
    out_image->height = height;
    out_image->pitch_bytes[0] = static_cast<int32_t>(pitch);
    out_image->format = output_format;
    out_image->cuda_stream = stream;
    output_slab->used_bytes = out_offset + bytes;
    output_slab->cuda_stream = stream;
    status = record_ready_event(output_slab, stream, &out_image->ready_event);
    return status;
}

RsAiStatus rs_ai_compress_device_to_slab(RsAiCompressionHandle handle,
                                         const RsAiDeviceSpan* input,
                                         RsAiGpuSlab* output_slab,
                                         void* cuda_stream,
                                         RsAiDeviceSpan* out_compressed)
{
    if (handle == nullptr || input == nullptr || input->ptr == nullptr || input->bytes == 0 ||
        output_slab == nullptr || out_compressed == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    CompressorHandle* h = static_cast<CompressorHandle*>(handle);
    if (h->codec != RS_AI_COMPRESSION_NVCOMP_LZ4 && h->codec != RS_AI_COMPRESSION_NVCOMP_GDEFLATE) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    cudaError_t cst = cudaSetDevice(h->cuda_device);
    if (cst != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }

    cudaStream_t stream = as_stream(cuda_stream != nullptr ? cuda_stream : output_slab->cuda_stream);
    try {
        size_t actual_len = 0;
        if (h->codec == RS_AI_COMPRESSION_NVCOMP_LZ4) {
            nvcomp::LZ4Manager manager(h->chunk_bytes, nvcompBatchedLZ4CompressDefaultOpts,
                                       nvcompBatchedLZ4DecompressDefaultOpts, stream);
            nvcomp::CompressionConfig cfg = manager.configure_compression(static_cast<size_t>(input->bytes));
            void* out_ptr = nullptr;
            uint64_t out_offset = 0;
            RsAiStatus status = reserve_slab_bytes(output_slab, cfg.max_compressed_buffer_size, &out_ptr, &out_offset);
            if (status != RS_AI_STATUS_OK) {
                return status;
            }
            manager.compress(static_cast<const uint8_t*>(input->ptr), static_cast<uint8_t*>(out_ptr), cfg);
            actual_len = manager.get_compressed_output_size(static_cast<const uint8_t*>(out_ptr));
            output_slab->used_bytes = out_offset + static_cast<uint64_t>(actual_len);
            out_compressed->ptr = out_ptr;
        } else {
            nvcomp::GdeflateManager manager(h->chunk_bytes, nvcompBatchedGdeflateCompressDefaultOpts,
                                            nvcompBatchedGdeflateDecompressDefaultOpts, stream);
            nvcomp::CompressionConfig cfg = manager.configure_compression(static_cast<size_t>(input->bytes));
            void* out_ptr = nullptr;
            uint64_t out_offset = 0;
            RsAiStatus status = reserve_slab_bytes(output_slab, cfg.max_compressed_buffer_size, &out_ptr, &out_offset);
            if (status != RS_AI_STATUS_OK) {
                return status;
            }
            manager.compress(static_cast<const uint8_t*>(input->ptr), static_cast<uint8_t*>(out_ptr), cfg);
            actual_len = manager.get_compressed_output_size(static_cast<const uint8_t*>(out_ptr));
            output_slab->used_bytes = out_offset + static_cast<uint64_t>(actual_len);
            out_compressed->ptr = out_ptr;
        }
        output_slab->cuda_stream = stream;
        out_compressed->bytes = static_cast<uint64_t>(actual_len);
        out_compressed->kind = RS_AI_BUFFER_CUDA_DEVICE;
        return record_ready_event(output_slab, stream, nullptr);
    } catch (...) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

RsAiStatus rs_ai_decompress_device_to_slab(RsAiCompressionHandle handle,
                                           const RsAiDeviceSpan* input,
                                           RsAiGpuSlab* output_slab,
                                           uint64_t expected_output_bytes,
                                           void* cuda_stream,
                                           RsAiDeviceSpan* out_decompressed)
{
    if (handle == nullptr || input == nullptr || input->ptr == nullptr || input->bytes == 0 ||
        output_slab == nullptr || out_decompressed == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    CompressorHandle* h = static_cast<CompressorHandle*>(handle);
    if (h->codec != RS_AI_COMPRESSION_NVCOMP_LZ4 && h->codec != RS_AI_COMPRESSION_NVCOMP_GDEFLATE) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    cudaError_t cst = cudaSetDevice(h->cuda_device);
    if (cst != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }

    cudaStream_t stream = as_stream(cuda_stream != nullptr ? cuda_stream : output_slab->cuda_stream);
    try {
        size_t out_bytes = 0;
        void* out_ptr = nullptr;
        uint64_t out_offset = 0;
        if (h->codec == RS_AI_COMPRESSION_NVCOMP_LZ4) {
            nvcomp::LZ4Manager manager(h->chunk_bytes, nvcompBatchedLZ4CompressDefaultOpts,
                                       nvcompBatchedLZ4DecompressDefaultOpts, stream);
            nvcomp::DecompressionConfig cfg =
                manager.configure_decompression(static_cast<const uint8_t*>(input->ptr));
            out_bytes = cfg.decomp_data_size;
            if (expected_output_bytes != 0 && expected_output_bytes != static_cast<uint64_t>(out_bytes)) {
                return RS_AI_STATUS_INVALID_ARGUMENT;
            }
            RsAiStatus status = reserve_slab_bytes(output_slab, out_bytes, &out_ptr, &out_offset);
            if (status != RS_AI_STATUS_OK) {
                return status;
            }
            manager.decompress(static_cast<uint8_t*>(out_ptr), static_cast<const uint8_t*>(input->ptr), cfg);
            cst = cudaStreamSynchronize(stream);
            if (cst != cudaSuccess) {
                return RS_AI_STATUS_RUNTIME_ERROR;
            }
        } else {
            nvcomp::GdeflateManager manager(h->chunk_bytes, nvcompBatchedGdeflateCompressDefaultOpts,
                                            nvcompBatchedGdeflateDecompressDefaultOpts, stream);
            nvcomp::DecompressionConfig cfg =
                manager.configure_decompression(static_cast<const uint8_t*>(input->ptr));
            out_bytes = cfg.decomp_data_size;
            if (expected_output_bytes != 0 && expected_output_bytes != static_cast<uint64_t>(out_bytes)) {
                return RS_AI_STATUS_INVALID_ARGUMENT;
            }
            RsAiStatus status = reserve_slab_bytes(output_slab, out_bytes, &out_ptr, &out_offset);
            if (status != RS_AI_STATUS_OK) {
                return status;
            }
            manager.decompress(static_cast<uint8_t*>(out_ptr), static_cast<const uint8_t*>(input->ptr), cfg);
            cst = cudaStreamSynchronize(stream);
            if (cst != cudaSuccess) {
                return RS_AI_STATUS_RUNTIME_ERROR;
            }
        }
        output_slab->used_bytes = out_offset + static_cast<uint64_t>(out_bytes);
        output_slab->cuda_stream = stream;
        out_decompressed->ptr = out_ptr;
        out_decompressed->bytes = static_cast<uint64_t>(out_bytes);
        out_decompressed->kind = RS_AI_BUFFER_CUDA_DEVICE;
        return record_ready_event(output_slab, stream, nullptr);
    } catch (...) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

RsAiStatus rs_ai_destroy_compressor(RsAiCompressionHandle handle)
{
    if (handle != nullptr) {
        CompressorHandle* h = static_cast<CompressorHandle*>(handle);
        if (h->jpeg_state != nullptr) {
            nvjpegJpegStateDestroy(h->jpeg_state);
        }
        if (h->encoder_params != nullptr) {
            nvjpegEncoderParamsDestroy(h->encoder_params);
        }
        if (h->encoder_state != nullptr) {
            nvjpegEncoderStateDestroy(h->encoder_state);
        }
        if (h->nvjpeg != nullptr) {
            nvjpegDestroy(h->nvjpeg);
        }
        delete h;
    }
    return RS_AI_STATUS_OK;
}

#ifndef HAVE_RS_AI_LIBTORCH
RsAiStatus rs_ai_extract_from_device_image_to_slab(RsAiHandle handle,
                                                   const RsAiDeviceImage* image,
                                                   uint32_t frame_id,
                                                   double pts_ms,
                                                   RsAiGpuSlab* output_slab,
                                                   RsAiDeviceFrame* out_frame)
{
    if (handle == nullptr || image == nullptr || image->plane_count < 1 || image->planes[0].ptr == nullptr ||
        image->width <= 0 || image->height <= 0 || output_slab == nullptr || out_frame == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    AiHandle* h = static_cast<AiHandle*>(handle);
    const int channels = image_channels(image->format);
    if (channels != 3 && channels != 4) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    cudaError_t st = cudaSetDevice(h->cuda_device);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }

    const uint64_t row_bytes = static_cast<uint64_t>(image->width) * static_cast<uint64_t>(channels);
    const uint64_t bytes = row_bytes * static_cast<uint64_t>(image->height);
    void* out_ptr = nullptr;
    uint64_t out_offset = 0;
    RsAiStatus status = reserve_slab_bytes(output_slab, bytes, &out_ptr, &out_offset);
    if (status != RS_AI_STATUS_OK) {
        return status;
    }
    cudaStream_t stream = as_stream(image->cuda_stream != nullptr ? image->cuda_stream : output_slab->cuda_stream);
    st = cudaMemcpy2DAsync(out_ptr, static_cast<size_t>(row_bytes),
                           image->planes[0].ptr,
                           static_cast<size_t>(image->pitch_bytes[0] > 0 ? image->pitch_bytes[0] : row_bytes),
                           static_cast<size_t>(row_bytes),
                           static_cast<size_t>(image->height),
                           cudaMemcpyDeviceToDevice,
                           stream);
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }

    output_slab->used_bytes = out_offset + bytes;
    output_slab->cuda_stream = stream;
    h->tensor_descs.resize(1);
    RsAiTensorDesc& tensor = h->tensor_descs[0];
    tensor = {};
    tensor.name = image->format == RS_AI_IMAGE_FORMAT_RGBA8 ? "visual_rgba" : "visual_bgr";
    tensor.group = "preview";
    tensor.role = "ai_c_abi_device_image";
    tensor.ndim = 3;
    tensor.shape[0] = image->height;
    tensor.shape[1] = image->width;
    tensor.shape[2] = channels;
    tensor.full_ndim = 3;
    tensor.full_shape[0] = image->height;
    tensor.full_shape[1] = image->width;
    tensor.full_shape[2] = channels;
    tensor.dtype = RS_AI_DTYPE_UINT8;
    tensor.encoding = RS_AI_ENCODING_RAW_MASK_U8;
    tensor.priority = RS_AI_PRIORITY_NORMAL;
    tensor.byte_offset = 0;
    tensor.byte_size = bytes;
    tensor.scale = 1.0f;
    tensor.zero_point = 0;

    *out_frame = {};
    out_frame->desc.protocol_version = RS_AI_GPU_DIRECT_PROTOCOL_VERSION;
    out_frame->desc.resource_mode = h->resource_mode;
    out_frame->desc.split_layer = static_cast<uint32_t>(h->split_layer);
    out_frame->desc.tensor_count = 1;
    out_frame->desc.frame_id = frame_id;
    out_frame->desc.pts_ms = pts_ms;
    out_frame->desc.source_h = image->height;
    out_frame->desc.source_w = image->width;
    out_frame->desc.network_h = h->image_size > 0 ? h->image_size : image->height;
    out_frame->desc.network_w = h->image_size > 0 ? h->image_size : image->width;
    out_frame->desc.payload_nbytes = bytes;
    out_frame->desc.tensors = h->tensor_descs.data();
    out_frame->device_payload = static_cast<const uint8_t*>(out_ptr);
    out_frame->device_payload_nbytes = bytes;
    out_frame->cuda_stream = stream;
    out_frame->slab_id = output_slab->slab_id;
    return record_ready_event(output_slab, stream, &out_frame->ready_event);
}
#endif

RsAiStatus rs_ai_baseband_tx_from_device_frame(RsAiBasebandHandle,
                                               const RsAiDeviceFrame*,
                                               RsAiGpuSlab*,
                                               RsAiDeviceSpan*)
{
    return RS_AI_STATUS_UNSUPPORTED;
}

RsAiStatus rs_ai_baseband_rx_to_device_llr(RsAiBasebandHandle,
                                           const RsAiDeviceSpan*,
                                           RsAiDeviceLlr*)
{
    return RS_AI_STATUS_UNSUPPORTED;
}

RsAiStatus rs_ai_bp_osd_decode_from_device_llr(RsAiBasebandHandle,
                                               const RsAiDeviceLlr*,
                                               RsAiGpuSlab*,
                                               RsAiDeviceFrame*)
{
    return RS_AI_STATUS_UNSUPPORTED;
}

RsAiStatus rs_ai_render_detections_to_texture(RsAiUiHandle handle,
                                              const RsAiDeviceFrame* preview_frame,
                                              const RsAiDetectionList* detections,
                                              RsAiGpuSlab* output_slab,
                                              RsAiUiTexture* out_texture)
{
    if (preview_frame == nullptr || preview_frame->device_payload == nullptr ||
        output_slab == nullptr || out_texture == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    const UiRendererHandle* h = static_cast<const UiRendererHandle*>(handle);
    if (h != nullptr) {
        cudaError_t st = cudaSetDevice(h->cuda_device);
        if (st != cudaSuccess) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
    }
    const int width = preview_frame->desc.source_w;
    const int height = preview_frame->desc.source_h;
    if (width <= 0 || height <= 0 || preview_frame->desc.tensor_count == 0 || preview_frame->desc.tensors == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    const RsAiTensorDesc& tensor = preview_frame->desc.tensors[0];
    const int channels = tensor.ndim >= 3 ? tensor.shape[2] : 3;
    if (channels != 3 && channels != 4) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    const int src_format = channels == 4 ? RS_AI_IMAGE_FORMAT_RGBA8 : RS_AI_IMAGE_FORMAT_BGR8;
    const uint64_t pitch = static_cast<uint64_t>(width) * 4ull;
    const uint64_t bytes = pitch * static_cast<uint64_t>(height);
    void* out_ptr = nullptr;
    uint64_t out_offset = 0;
    RsAiStatus status = reserve_slab_bytes(output_slab, bytes, &out_ptr, &out_offset);
    if (status != RS_AI_STATUS_OK) {
        return status;
    }

    cudaStream_t stream = as_stream(preview_frame->cuda_stream != nullptr ? preview_frame->cuda_stream : output_slab->cuda_stream);
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    image_to_rgba_kernel<<<grid, block, 0, stream>>>(
        preview_frame->device_payload,
        width,
        height,
        width * channels,
        channels,
        src_format,
        static_cast<uint8_t*>(out_ptr),
        static_cast<int>(pitch));
    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }

    RsAiDetection* d_detections = nullptr;
    if (detections != nullptr && detections->count > 0 && detections->detections != nullptr) {
        const size_t det_bytes = sizeof(RsAiDetection) * static_cast<size_t>(detections->count);
        st = cudaMalloc(&d_detections, det_bytes);
        if (st != cudaSuccess) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
        st = cudaMemcpyAsync(d_detections, detections->detections, det_bytes, cudaMemcpyHostToDevice, stream);
        if (st != cudaSuccess) {
            cudaFree(d_detections);
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
        const dim3 det_grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, detections->count);
        draw_detections_kernel<<<det_grid, block, 0, stream>>>(
            static_cast<uint8_t*>(out_ptr),
            width,
            height,
            static_cast<int>(pitch),
            d_detections,
            detections->count);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            cudaFree(d_detections);
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
        st = cudaStreamSynchronize(stream);
        cudaFree(d_detections);
        if (st != cudaSuccess) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
    }

    output_slab->used_bytes = out_offset + bytes;
    output_slab->cuda_stream = stream;
    *out_texture = {};
    out_texture->rgba.ptr = out_ptr;
    out_texture->rgba.bytes = bytes;
    out_texture->rgba.kind = RS_AI_BUFFER_CUDA_DEVICE;
    out_texture->width = width;
    out_texture->height = height;
    out_texture->pitch_bytes = static_cast<int32_t>(pitch);
    out_texture->cuda_stream = stream;
    return record_ready_event(output_slab, stream, &out_texture->ready_event);
}

}  // extern "C"
