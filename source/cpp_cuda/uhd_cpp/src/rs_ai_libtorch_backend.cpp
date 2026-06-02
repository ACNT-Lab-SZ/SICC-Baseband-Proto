#include "rs_ai_gpu_direct_api.h"

#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct LibTorchAiHandle {
    std::string model_path;
    RsAiResourceMode resource_mode = RS_AI_RESOURCE_UNIFORM;
    int32_t split_layer = 0;
    int32_t image_size = 0;
    int32_t cuda_device = 0;
    torch::jit::script::Module module;
    std::vector<RsAiTensorDesc> tensor_descs;
    std::vector<RsAiDetection> detections;
    std::unordered_map<const void*, torch::Tensor> live_payloads;
};

RsAiStatus cuda_status(cudaError_t status)
{
    return status == cudaSuccess ? RS_AI_STATUS_OK : RS_AI_STATUS_RUNTIME_ERROR;
}

torch::Tensor first_tensor_from_ivalue(const torch::jit::IValue& value)
{
    if (value.isTensor()) {
        return value.toTensor();
    }
    if (value.isTuple()) {
        for (const auto& item : value.toTuple()->elements()) {
            torch::Tensor t = first_tensor_from_ivalue(item);
            if (t.defined()) {
                return t;
            }
        }
    }
    if (value.isList()) {
        const auto list = value.toList();
        for (const auto& item : list) {
            torch::Tensor t = first_tensor_from_ivalue(item);
            if (t.defined()) {
                return t;
            }
        }
    }
    return {};
}

RsAiTensorDType dtype_from_torch(const torch::Tensor& tensor)
{
    switch (tensor.scalar_type()) {
    case torch::kInt8:
        return RS_AI_DTYPE_INT8;
    case torch::kUInt8:
        return RS_AI_DTYPE_UINT8;
    case torch::kFloat16:
        return RS_AI_DTYPE_FLOAT16;
    case torch::kFloat32:
        return RS_AI_DTYPE_FLOAT32;
    default:
        return RS_AI_DTYPE_FLOAT32;
    }
}

RsAiTensorEncoding encoding_from_torch(const torch::Tensor& tensor)
{
    switch (tensor.scalar_type()) {
    case torch::kInt8:
        return RS_AI_ENCODING_RAW_SYMMETRIC_INT8;
    case torch::kUInt8:
        return RS_AI_ENCODING_RAW_MASK_U8;
    case torch::kFloat16:
        return RS_AI_ENCODING_RAW_FLOAT16;
    case torch::kFloat32:
    default:
        return RS_AI_ENCODING_RAW_FLOAT32;
    }
}

void fill_tensor_desc(RsAiTensorDesc& desc, const torch::Tensor& tensor)
{
    desc = {};
    desc.name = "libtorch_output_0";
    desc.group = "ai";
    desc.role = "libtorch_forward_output";
    desc.ndim = static_cast<uint32_t>(std::min<int64_t>(tensor.dim(), RS_AI_MAX_TENSOR_DIMS));
    desc.full_ndim = desc.ndim;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        desc.shape[i] = static_cast<int32_t>(tensor.size(i));
        desc.full_shape[i] = desc.shape[i];
    }
    desc.dtype = dtype_from_torch(tensor);
    desc.encoding = encoding_from_torch(tensor);
    desc.priority = RS_AI_PRIORITY_NORMAL;
    desc.byte_offset = 0;
    desc.byte_size = static_cast<uint64_t>(tensor.nbytes());
    desc.scale = 1.0f;
    desc.zero_point = 0;
}

torch::Tensor make_input_tensor(const uint8_t* device_bgr,
                                int32_t height,
                                int32_t width,
                                int32_t stride_bytes,
                                int32_t image_size,
                                int32_t cuda_device)
{
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, cuda_device);
    const int64_t stride0 = stride_bytes > 0 ? stride_bytes : static_cast<int64_t>(width) * 3;
    torch::Tensor view = torch::from_blob(
        const_cast<uint8_t*>(device_bgr),
        {height, width, 3},
        {stride0, 3, 1},
        options);
    torch::Tensor input = view.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div_(255.0);
    if (image_size > 0 && (image_size != height || image_size != width)) {
        input = torch::nn::functional::interpolate(
            input,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{image_size, image_size})
                .mode(torch::kBilinear)
                .align_corners(false));
    }
    return input.contiguous();
}

RsAiStatus forward_to_frame(LibTorchAiHandle* h,
                            const uint8_t* device_bgr,
                            int32_t height,
                            int32_t width,
                            int32_t stride_bytes,
                            uint32_t frame_id,
                            double pts_ms,
                            void* cuda_stream,
                            RsAiDeviceFrame* out_frame)
{
    if (h == nullptr || device_bgr == nullptr || height <= 0 || width <= 0 || out_frame == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t cst = cudaSetDevice(h->cuda_device);
    if (cst != cudaSuccess) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
    if (cuda_stream != nullptr) {
        cst = cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream));
        if (cst != cudaSuccess) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
    }

    try {
        torch::NoGradGuard no_grad;
        torch::Tensor input = make_input_tensor(device_bgr, height, width, stride_bytes, h->image_size, h->cuda_device);
        torch::jit::IValue output = h->module.forward({input});
        torch::Tensor payload = first_tensor_from_ivalue(output);
        if (!payload.defined()) {
            return RS_AI_STATUS_UNSUPPORTED;
        }
        if (!payload.is_cuda()) {
            payload = payload.to(torch::Device(torch::kCUDA, h->cuda_device));
        }
        payload = payload.contiguous();
        cudaDeviceSynchronize();

        h->tensor_descs.resize(1);
        fill_tensor_desc(h->tensor_descs[0], payload);
        const void* key = payload.data_ptr();
        h->live_payloads[key] = payload;

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
        out_frame->desc.payload_nbytes = static_cast<uint64_t>(payload.nbytes());
        out_frame->desc.tensors = h->tensor_descs.data();
        out_frame->device_payload = static_cast<const uint8_t*>(payload.data_ptr());
        out_frame->device_payload_nbytes = static_cast<uint64_t>(payload.nbytes());
        out_frame->cuda_stream = nullptr;
        out_frame->slab_id = 0xA17C0001u;
        return RS_AI_STATUS_OK;
    } catch (const std::exception&) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

}  // namespace

extern "C" {

RsAiStatus rs_ai_create(const char* model_path,
                        RsAiResourceMode resource_mode,
                        int32_t split_layer,
                        int32_t image_size,
                        int32_t cuda_device,
                        RsAiHandle* out_handle)
{
    if (model_path == nullptr || out_handle == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;
    try {
        cudaError_t cst = cudaSetDevice(cuda_device);
        if (cst != cudaSuccess) {
            return RS_AI_STATUS_RUNTIME_ERROR;
        }
        auto h = std::make_unique<LibTorchAiHandle>();
        h->model_path = model_path;
        h->resource_mode = resource_mode;
        h->split_layer = split_layer;
        h->image_size = image_size;
        h->cuda_device = cuda_device;
        h->module = torch::jit::load(model_path, torch::Device(torch::kCUDA, cuda_device));
        h->module.eval();
        *out_handle = h.release();
        return RS_AI_STATUS_OK;
    } catch (const std::exception&) {
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

RsAiStatus rs_ai_destroy(RsAiHandle handle)
{
    delete static_cast<LibTorchAiHandle*>(handle);
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
    return forward_to_frame(
        static_cast<LibTorchAiHandle*>(handle),
        device_bgr,
        height,
        width,
        stride_bytes,
        frame_id,
        pts_ms,
        cuda_stream,
        out_frame);
}

RsAiStatus rs_ai_extract_from_device_image_to_slab(RsAiHandle handle,
                                                   const RsAiDeviceImage* image,
                                                   uint32_t frame_id,
                                                   double pts_ms,
                                                   RsAiGpuSlab*,
                                                   RsAiDeviceFrame* out_frame)
{
    if (image == nullptr || image->plane_count < 1 || image->planes[0].ptr == nullptr ||
        image->format != RS_AI_IMAGE_FORMAT_BGR8) {
        return RS_AI_STATUS_UNSUPPORTED;
    }
    return forward_to_frame(
        static_cast<LibTorchAiHandle*>(handle),
        static_cast<const uint8_t*>(image->planes[0].ptr),
        image->height,
        image->width,
        image->pitch_bytes[0],
        frame_id,
        pts_ms,
        image->cuda_stream,
        out_frame);
}

RsAiStatus rs_ai_detect_from_device_frame(RsAiHandle handle,
                                          const RsAiDeviceFrame* frame,
                                          float conf_threshold,
                                          float,
                                          RsAiDetectionList* out_detections)
{
    if (handle == nullptr || frame == nullptr || frame->device_payload == nullptr ||
        frame->desc.tensor_count == 0 || frame->desc.tensors == nullptr || out_detections == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    LibTorchAiHandle* h = static_cast<LibTorchAiHandle*>(handle);
    h->detections.clear();
    const RsAiTensorDesc& desc = frame->desc.tensors[0];
    if (desc.dtype != RS_AI_DTYPE_FLOAT32 || desc.ndim < 2 || desc.shape[desc.ndim - 1] < 6) {
        const bool is_yolo_chw = desc.ndim == 3 && desc.shape[0] == 1 && desc.shape[1] >= 5;
        if (!is_yolo_chw) {
            *out_detections = {};
            return RS_AI_STATUS_OK;
        }
    }
    try {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, h->cuda_device);
        if (desc.ndim == 3 && desc.shape[0] == 1 && desc.shape[1] >= 5) {
            const int64_t channels = desc.shape[1];
            const int64_t anchors = desc.shape[2];
            torch::Tensor yolo = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(frame->device_payload)),
                {1, channels, anchors},
                options).squeeze(0).transpose(0, 1).contiguous().to(torch::kCPU);
            const float* data = yolo.data_ptr<float>();
            for (int64_t r = 0; r < anchors; ++r) {
                if (h->detections.size() >= 1024) {
                    break;
                }
                const float* row = data + r * channels;
                int class_id = 0;
                float score = row[4];
                if (channels > 5) {
                    score = 0.0f;
                    for (int64_t c = 4; c < channels; ++c) {
                        if (row[c] > score) {
                            score = row[c];
                            class_id = static_cast<int>(c - 4);
                        }
                    }
                }
                if (score < conf_threshold) {
                    continue;
                }
                const float cx = row[0];
                const float cy = row[1];
                const float w = row[2];
                const float ht = row[3];
                RsAiDetection det{};
                det.frame_id = frame->desc.frame_id;
                det.x1 = cx - 0.5f * w;
                det.y1 = cy - 0.5f * ht;
                det.x2 = cx + 0.5f * w;
                det.y2 = cy + 0.5f * ht;
                det.confidence = score;
                det.class_id = class_id;
                h->detections.push_back(det);
            }
        } else {
            int64_t rows = 1;
            for (uint32_t i = 0; i + 1 < desc.ndim; ++i) {
                rows *= std::max(1, desc.shape[i]);
            }
            const int64_t cols = desc.shape[desc.ndim - 1];
            torch::Tensor det_tensor = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(frame->device_payload)),
                {rows, cols},
                options).to(torch::kCPU);
            const float* data = det_tensor.data_ptr<float>();
            for (int64_t r = 0; r < rows; ++r) {
                if (h->detections.size() >= 1024) {
                    break;
                }
                const float* row = data + r * cols;
                const float conf = row[4];
                if (conf < conf_threshold) {
                    continue;
                }
                RsAiDetection det{};
                det.frame_id = frame->desc.frame_id;
                det.x1 = row[0];
                det.y1 = row[1];
                det.x2 = row[2];
                det.y2 = row[3];
                det.confidence = conf;
                det.class_id = static_cast<int32_t>(row[5]);
                h->detections.push_back(det);
            }
        }
        out_detections->count = static_cast<uint32_t>(h->detections.size());
        out_detections->detections = h->detections.data();
        return RS_AI_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "rs_ai_detect_from_device_frame LibTorch error: " << e.what() << "\n";
        return RS_AI_STATUS_RUNTIME_ERROR;
    }
}

RsAiStatus rs_ai_release_device_frame(RsAiHandle handle, RsAiDeviceFrame* frame)
{
    if (handle == nullptr || frame == nullptr) {
        return RS_AI_STATUS_INVALID_ARGUMENT;
    }
    LibTorchAiHandle* h = static_cast<LibTorchAiHandle*>(handle);
    if (frame->device_payload != nullptr) {
        h->live_payloads.erase(frame->device_payload);
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

}  // extern "C"
