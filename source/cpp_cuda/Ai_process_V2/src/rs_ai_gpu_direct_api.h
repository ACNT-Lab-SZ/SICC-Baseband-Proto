#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RS_AI_GPU_DIRECT_PROTOCOL_VERSION 4u
#define RS_AI_MAX_TENSOR_DIMS 8u
#define RS_AI_MAX_IMAGE_PLANES 4u

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
    RS_AI_ENCODING_RAW_MASK_U8 = 4,
    RS_AI_ENCODING_NVJPEG = 5,
    RS_AI_ENCODING_NVCOMP_LZ4 = 6,
    RS_AI_ENCODING_NVCOMP_GDEFLATE = 7
} RsAiTensorEncoding;

typedef enum RsAiTensorPriority {
    RS_AI_PRIORITY_NORMAL = 0,
    RS_AI_PRIORITY_LOW = 1,
    RS_AI_PRIORITY_HIGH = 2,
    RS_AI_PRIORITY_CONTROL = 3
} RsAiTensorPriority;

typedef enum RsAiDeviceBufferKind {
    RS_AI_BUFFER_UNKNOWN = 0,
    RS_AI_BUFFER_CUDA_DEVICE = 1,
    RS_AI_BUFFER_CUDA_ARRAY = 2,
    RS_AI_BUFFER_EXTERNAL_DEVICE = 3
} RsAiDeviceBufferKind;

typedef enum RsAiVideoCodec {
    RS_AI_VIDEO_CODEC_NONE = 0,
    RS_AI_VIDEO_CODEC_H264 = 1,
    RS_AI_VIDEO_CODEC_HEVC = 2,
    RS_AI_VIDEO_CODEC_AV1 = 3
} RsAiVideoCodec;

typedef enum RsAiImageFormat {
    RS_AI_IMAGE_FORMAT_UNKNOWN = 0,
    RS_AI_IMAGE_FORMAT_BGR8 = 1,
    RS_AI_IMAGE_FORMAT_RGB8 = 2,
    RS_AI_IMAGE_FORMAT_RGBA8 = 3,
    RS_AI_IMAGE_FORMAT_NV12 = 4,
    RS_AI_IMAGE_FORMAT_YUV420P = 5
} RsAiImageFormat;

typedef enum RsAiCompressionCodec {
    RS_AI_COMPRESSION_NONE = 0,
    RS_AI_COMPRESSION_NVJPEG = 1,
    RS_AI_COMPRESSION_NVCOMP_LZ4 = 2,
    RS_AI_COMPRESSION_NVCOMP_GDEFLATE = 3
} RsAiCompressionCodec;

typedef struct RsAiDeviceSpan {
    void* ptr;
    uint64_t bytes;
    RsAiDeviceBufferKind kind;
} RsAiDeviceSpan;

typedef struct RsAiGpuSlab {
    RsAiDeviceSpan memory;
    uint64_t capacity_bytes;
    uint64_t used_bytes;
    uint32_t alignment_bytes;
    uint32_t slab_id;
    void* cuda_stream;
    void* ready_event;
} RsAiGpuSlab;

typedef struct RsAiDeviceImage {
    RsAiDeviceSpan planes[RS_AI_MAX_IMAGE_PLANES];
    uint32_t plane_count;
    int32_t width;
    int32_t height;
    int32_t pitch_bytes[RS_AI_MAX_IMAGE_PLANES];
    RsAiImageFormat format;
    void* cuda_stream;
    void* ready_event;
} RsAiDeviceImage;

typedef struct RsAiDeviceLlr {
    const float* device_llr;
    uint32_t blocks;
    uint32_t code_n;
    uint32_t code_k;
    void* cuda_stream;
    void* ready_event;
} RsAiDeviceLlr;

typedef struct RsAiUiTexture {
    RsAiDeviceSpan rgba;
    int32_t width;
    int32_t height;
    int32_t pitch_bytes;
    void* cuda_stream;
    void* ready_event;
} RsAiUiTexture;

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
    uint64_t device_payload_nbytes;
    void* cuda_stream;
    void* ready_event;
    uint32_t slab_id;
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
typedef void* RsAiVideoCodecHandle;
typedef void* RsAiCompressionHandle;
typedef void* RsAiBasebandHandle;
typedef void* RsAiUiHandle;

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

RsAiStatus rs_ai_create_ui_renderer(int32_t cuda_device, RsAiUiHandle* out_handle);

RsAiStatus rs_ai_destroy_ui_renderer(RsAiUiHandle handle);

/*
 * GPU-native pipeline extension.
 *
 * All pointers in the following API are CUDA device pointers unless explicitly
 * documented otherwise. Implementations must preserve stream ordering:
 * consumers either enqueue work on the producer stream or wait on ready_event.
 */

RsAiStatus rs_ai_allocate_slab(uint64_t capacity_bytes,
                               uint32_t alignment_bytes,
                               int32_t cuda_device,
                               void* cuda_stream,
                               RsAiGpuSlab* out_slab);

RsAiStatus rs_ai_wrap_external_slab(void* device_ptr,
                                    uint64_t capacity_bytes,
                                    uint32_t alignment_bytes,
                                    uint32_t slab_id,
                                    void* cuda_stream,
                                    RsAiGpuSlab* out_slab);

RsAiStatus rs_ai_release_slab(RsAiGpuSlab* slab);

RsAiStatus rs_ai_create_video_decoder(RsAiVideoCodec codec,
                                      int32_t cuda_device,
                                      RsAiVideoCodecHandle* out_handle);

RsAiStatus rs_ai_nvdec_decode_to_device(RsAiVideoCodecHandle handle,
                                        const uint8_t* host_bitstream,
                                        uint64_t host_bitstream_bytes,
                                        RsAiGpuSlab* output_slab,
                                        void* cuda_stream,
                                        RsAiDeviceImage* out_image);

RsAiStatus rs_ai_create_video_encoder(RsAiVideoCodec codec,
                                      int32_t cuda_device,
                                      RsAiVideoCodecHandle* out_handle);

RsAiStatus rs_ai_nvenc_encode_from_device(RsAiVideoCodecHandle handle,
                                          const RsAiDeviceImage* image,
                                          uint8_t* host_bitstream,
                                          uint64_t host_capacity_bytes,
                                          uint64_t* out_bytes,
                                          void* cuda_stream);

RsAiStatus rs_ai_destroy_video_codec(RsAiVideoCodecHandle handle);

RsAiStatus rs_ai_create_compressor(RsAiCompressionCodec codec,
                                   int32_t cuda_device,
                                   RsAiCompressionHandle* out_handle);

RsAiStatus rs_ai_nvjpeg_encode_image_to_slab(RsAiCompressionHandle handle,
                                             const RsAiDeviceImage* image,
                                             int32_t quality,
                                             RsAiGpuSlab* output_slab,
                                             void* cuda_stream,
                                             RsAiDeviceSpan* out_jpeg);

RsAiStatus rs_ai_nvjpeg_decode_host_to_slab(RsAiCompressionHandle handle,
                                            const uint8_t* host_jpeg,
                                            uint64_t host_jpeg_bytes,
                                            RsAiImageFormat output_format,
                                            RsAiGpuSlab* output_slab,
                                            void* cuda_stream,
                                            RsAiDeviceImage* out_image);

RsAiStatus rs_ai_compress_device_to_slab(RsAiCompressionHandle handle,
                                         const RsAiDeviceSpan* input,
                                         RsAiGpuSlab* output_slab,
                                         void* cuda_stream,
                                         RsAiDeviceSpan* out_compressed);

RsAiStatus rs_ai_decompress_device_to_slab(RsAiCompressionHandle handle,
                                           const RsAiDeviceSpan* input,
                                           RsAiGpuSlab* output_slab,
                                           uint64_t expected_output_bytes,
                                           void* cuda_stream,
                                           RsAiDeviceSpan* out_decompressed);

RsAiStatus rs_ai_destroy_compressor(RsAiCompressionHandle handle);

RsAiStatus rs_ai_extract_from_device_image_to_slab(RsAiHandle handle,
                                                   const RsAiDeviceImage* image,
                                                   uint32_t frame_id,
                                                   double pts_ms,
                                                   RsAiGpuSlab* output_slab,
                                                   RsAiDeviceFrame* out_frame);

RsAiStatus rs_ai_baseband_tx_from_device_frame(RsAiBasebandHandle handle,
                                               const RsAiDeviceFrame* frame,
                                               RsAiGpuSlab* iq_output_slab,
                                               RsAiDeviceSpan* out_iq_samples);

RsAiStatus rs_ai_baseband_rx_to_device_llr(RsAiBasebandHandle handle,
                                           const RsAiDeviceSpan* iq_samples,
                                           RsAiDeviceLlr* out_llr);

RsAiStatus rs_ai_bp_osd_decode_from_device_llr(RsAiBasebandHandle handle,
                                               const RsAiDeviceLlr* llr,
                                               RsAiGpuSlab* output_slab,
                                               RsAiDeviceFrame* out_frame);

RsAiStatus rs_ai_render_detections_to_texture(RsAiUiHandle handle,
                                              const RsAiDeviceFrame* preview_frame,
                                              const RsAiDetectionList* detections,
                                              RsAiGpuSlab* output_slab,
                                              RsAiUiTexture* out_texture);

#ifdef __cplusplus
}
#endif
