#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RS_AI_GPU_DIRECT_PROTOCOL_VERSION 2u
#define RS_AI_MAX_TENSOR_DIMS 8u

typedef enum RsAiStatus {
    RS_AI_STATUS_OK = 0,
    RS_AI_STATUS_INVALID_ARGUMENT = 1,
    RS_AI_STATUS_RUNTIME_ERROR = 2,
    RS_AI_STATUS_UNSUPPORTED = 3
} RsAiStatus;

typedef enum RsAiResourceMode {
    RS_AI_RESOURCE_UNIFORM = 1,
    RS_AI_RESOURCE_ROI_LAYERED = 2
} RsAiResourceMode;

typedef enum RsAiTensorDType {
    RS_AI_DTYPE_INT8 = 1,
    RS_AI_DTYPE_FLOAT16 = 2,
    RS_AI_DTYPE_FLOAT32 = 3,
    RS_AI_DTYPE_UINT8 = 4
} RsAiTensorDType;

typedef enum RsAiTensorEncoding {
    RS_AI_ENCODING_RAW_SYMMETRIC_INT8 = 1,
    RS_AI_ENCODING_RAW_FLOAT16 = 2,
    RS_AI_ENCODING_RAW_FLOAT32 = 3,
    RS_AI_ENCODING_RAW_MASK_U8 = 4
} RsAiTensorEncoding;

typedef enum RsAiTensorPriority {
    RS_AI_PRIORITY_NORMAL = 0,
    RS_AI_PRIORITY_LOW = 1,
    RS_AI_PRIORITY_HIGH = 2,
    RS_AI_PRIORITY_CONTROL = 3
} RsAiTensorPriority;

typedef struct RsAiTensorDesc {
    const char* name;
    const char* group;
    const char* role;
    uint32_t ndim;
    int32_t shape[RS_AI_MAX_TENSOR_DIMS];
    uint32_t full_ndim;
    int32_t full_shape[RS_AI_MAX_TENSOR_DIMS];
    RsAiTensorDType dtype;
    RsAiTensorEncoding encoding;
    RsAiTensorPriority priority;
    uint64_t byte_offset;
    uint64_t byte_size;
    float scale;
    int32_t zero_point;
} RsAiTensorDesc;

typedef struct RsAiFrameDesc {
    uint32_t protocol_version;
    RsAiResourceMode resource_mode;
    uint32_t split_layer;
    uint32_t tensor_count;
    uint32_t frame_id;
    double pts_ms;
    int32_t source_h;
    int32_t source_w;
    int32_t network_h;
    int32_t network_w;
    uint64_t payload_nbytes;
    const RsAiTensorDesc* tensors;
} RsAiFrameDesc;

typedef struct RsAiDeviceFrame {
    RsAiFrameDesc desc;
    const uint8_t* device_payload;
    void* cuda_stream;
} RsAiDeviceFrame;

typedef struct RsAiDetection {
    uint32_t frame_id;
    int32_t class_id;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
} RsAiDetection;

typedef struct RsAiDetectionList {
    uint32_t count;
    const RsAiDetection* detections;
} RsAiDetectionList;

/*
 * Baseline C/C++ ABI expected by the C++ baseband adapter.
 *
 * TX:
 *   AI module produces RsAiDeviceFrame.
 *   Baseband reads frame.device_payload directly on GPU and transmits exactly
 *   frame.desc.payload_nbytes bytes. The baseband should keep desc as metadata.
 *
 * RX:
 *   Baseband returns the received device pointer with the same RsAiFrameDesc.
 *   AI module consumes RsAiDeviceFrame and runs the receiver-side detector.
 *
 * The current repository provides a validated Python/Torch implementation of
 * the producer/consumer behavior. A production shared library can implement
 * these declarations with TensorRT, LibTorch, or a Python embedding bridge.
 */
typedef void* RsAiHandle;

RsAiStatus rs_ai_create(const char* model_path,
                        RsAiResourceMode resource_mode,
                        int32_t split_layer,
                        int32_t image_size,
                        int32_t cuda_device,
                        RsAiHandle* out_handle);

RsAiStatus rs_ai_destroy(RsAiHandle handle);

RsAiStatus rs_ai_extract_from_bgr_device(RsAiHandle handle,
                                         const uint8_t* device_bgr,
                                         int32_t height,
                                         int32_t width,
                                         int32_t stride_bytes,
                                         uint32_t frame_id,
                                         double pts_ms,
                                         void* cuda_stream,
                                         RsAiDeviceFrame* out_frame);

RsAiStatus rs_ai_detect_from_device_frame(RsAiHandle handle,
                                          const RsAiDeviceFrame* frame,
                                          float conf_threshold,
                                          float iou_threshold,
                                          RsAiDetectionList* out_detections);

RsAiStatus rs_ai_release_device_frame(RsAiHandle handle, RsAiDeviceFrame* frame);

RsAiStatus rs_ai_release_detections(RsAiHandle handle, RsAiDetectionList* detections);

#ifdef __cplusplus
}
#endif
