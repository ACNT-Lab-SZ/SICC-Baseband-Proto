#include "rs_ai_gpu_direct_api.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
        std::exit(2);
    }
}

void check_rs(RsAiStatus status, const char* what)
{
    if (status != RS_AI_STATUS_OK) {
        std::cerr << what << ": status=" << static_cast<int>(status) << "\n";
        std::exit(3);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::cerr << "rs_ai_gpu_direct_runtime_smoke begin argc=" << argc << "\n";
    constexpr int width = 64;
    constexpr int height = 48;
    constexpr int channels = 3;
    constexpr int pitch = width * channels;

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    std::vector<std::uint8_t> host(static_cast<size_t>(height * pitch), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y * pitch + x * channels);
            host[idx + 0] = static_cast<std::uint8_t>((x * 3) & 0xff);
            host[idx + 1] = static_cast<std::uint8_t>((y * 5) & 0xff);
            host[idx + 2] = static_cast<std::uint8_t>(((x + y) * 2) & 0xff);
        }
    }

    void* d_image = nullptr;
    check_cuda(cudaMalloc(&d_image, host.size()), "cudaMalloc image");
    check_cuda(cudaMemcpyAsync(d_image, host.data(), host.size(), cudaMemcpyHostToDevice, stream), "copy image");
    std::cerr << "stage image_alloc_ok\n";

    RsAiDeviceImage image{};
    image.planes[0].ptr = d_image;
    image.planes[0].bytes = host.size();
    image.planes[0].kind = RS_AI_BUFFER_CUDA_DEVICE;
    image.plane_count = 1;
    image.width = width;
    image.height = height;
    image.pitch_bytes[0] = pitch;
    image.format = RS_AI_IMAGE_FORMAT_BGR8;
    image.cuda_stream = stream;

    RsAiCompressionHandle codec = nullptr;
    check_rs(rs_ai_create_compressor(RS_AI_COMPRESSION_NVJPEG, 0, &codec), "create nvjpeg");

    RsAiGpuSlab jpeg_slab{};
    check_rs(rs_ai_allocate_slab(1u << 20, 256, 0, stream, &jpeg_slab), "allocate jpeg slab");
    RsAiDeviceSpan jpeg{};
    check_rs(rs_ai_nvjpeg_encode_image_to_slab(codec, &image, 85, &jpeg_slab, stream, &jpeg), "encode nvjpeg");
    check_cuda(cudaStreamSynchronize(stream), "sync encode");

    std::vector<std::uint8_t> host_jpeg(static_cast<size_t>(jpeg.bytes), 0);
    check_cuda(cudaMemcpy(host_jpeg.data(), jpeg.ptr, host_jpeg.size(), cudaMemcpyDeviceToHost), "copy jpeg");

    RsAiGpuSlab decode_slab{};
    check_rs(rs_ai_allocate_slab(width * height * channels * 2, 256, 0, stream, &decode_slab), "allocate decode slab");
    RsAiDeviceImage decoded{};
    check_rs(
        rs_ai_nvjpeg_decode_host_to_slab(
            codec,
            host_jpeg.data(),
            host_jpeg.size(),
            RS_AI_IMAGE_FORMAT_BGR8,
            &decode_slab,
            stream,
            &decoded),
        "decode nvjpeg");
    check_cuda(cudaStreamSynchronize(stream), "sync decode");
    std::cerr << "stage nvjpeg_roundtrip_ok\n";

    if (decoded.width != width || decoded.height != height || decoded.planes[0].ptr == nullptr) {
        std::cerr << "decoded image metadata mismatch\n";
        return 4;
    }

    auto roundtrip_nvcomp = [&](RsAiCompressionCodec comp_codec, const char* name) {
        RsAiCompressionHandle comp = nullptr;
        check_rs(rs_ai_create_compressor(comp_codec, 0, &comp), name);

        RsAiDeviceSpan raw{};
        raw.ptr = d_image;
        raw.bytes = host.size();
        raw.kind = RS_AI_BUFFER_CUDA_DEVICE;

        RsAiGpuSlab comp_slab{};
        check_rs(rs_ai_allocate_slab(host.size() * 2 + 4096, 256, 0, stream, &comp_slab), "allocate nvcomp slab");
        RsAiDeviceSpan compressed{};
        check_rs(rs_ai_compress_device_to_slab(comp, &raw, &comp_slab, stream, &compressed), "nvcomp compress");
        check_cuda(cudaStreamSynchronize(stream), "sync nvcomp compress");

        RsAiGpuSlab raw_slab{};
        check_rs(rs_ai_allocate_slab(host.size() + 4096, 256, 0, stream, &raw_slab), "allocate nvcomp decode slab");
        RsAiDeviceSpan restored{};
        check_rs(
            rs_ai_decompress_device_to_slab(comp, &compressed, &raw_slab, host.size(), stream, &restored),
            "nvcomp decompress");
        check_cuda(cudaStreamSynchronize(stream), "sync nvcomp decompress");

        std::vector<std::uint8_t> restored_host(host.size(), 0);
        check_cuda(cudaMemcpy(restored_host.data(), restored.ptr, restored_host.size(), cudaMemcpyDeviceToHost), "copy restored");
        if (restored.bytes != host.size() || std::memcmp(restored_host.data(), host.data(), host.size()) != 0) {
            std::cerr << name << " roundtrip mismatch\n";
            std::exit(5);
        }
        std::cout << name << "_smoke_ok compressed_bytes=" << compressed.bytes << "\n";
        rs_ai_release_slab(&raw_slab);
        rs_ai_release_slab(&comp_slab);
        rs_ai_destroy_compressor(comp);
    };

    roundtrip_nvcomp(RS_AI_COMPRESSION_NVCOMP_LZ4, "nvcomp_lz4");
    roundtrip_nvcomp(RS_AI_COMPRESSION_NVCOMP_GDEFLATE, "nvcomp_gdeflate");
    std::cerr << "stage nvcomp_roundtrip_ok\n";

    const char* ai_model = argc > 1 ? argv[1] : "mock-gpu-ai-c-abi";
    const std::string ai_model_path(ai_model);
    const bool yolo_torchscript =
        ai_model_path.find("yolo") != std::string::npos ||
        ai_model_path.find("viso") != std::string::npos;
    const int ai_image_size = yolo_torchscript ? 640 : width;
    const float ai_conf_threshold = yolo_torchscript ? 0.001f : 0.25f;
    RsAiHandle ai = nullptr;
    std::cerr << "stage ai_create_begin model=" << ai_model << " image_size=" << ai_image_size << "\n";
    check_rs(rs_ai_create(ai_model, RS_AI_RESOURCE_ROI_LAYERED, 22, ai_image_size, 0, &ai), "create ai c abi");
    std::cerr << "stage ai_create_ok\n";

    RsAiGpuSlab ai_slab{};
    check_rs(rs_ai_allocate_slab(host.size() + 4096, 256, 0, stream, &ai_slab), "allocate ai slab");
    RsAiDeviceFrame device_frame{};
    check_rs(
        rs_ai_extract_from_device_image_to_slab(ai, &image, 7, 123.0, &ai_slab, &device_frame),
        "ai extract device image");
    check_cuda(cudaStreamSynchronize(stream), "sync ai extract");
    std::cerr << "stage ai_extract_ok payload=" << device_frame.device_payload_nbytes << "\n";

    RsAiDetectionList detections{};
    check_rs(rs_ai_detect_from_device_frame(ai, &device_frame, ai_conf_threshold, 0.45f, &detections), "ai detect frame");
    std::cerr << "stage ai_detect_ok detections=" << detections.count << "\n";
    if (detections.count == 0 || detections.detections == nullptr) {
        if (!yolo_torchscript) {
            std::cerr << "ai c abi produced no detections\n";
            return 6;
        }
    }

    RsAiUiHandle ui = nullptr;
    check_rs(rs_ai_create_ui_renderer(0, &ui), "create ui renderer");
    RsAiGpuSlab texture_slab{};
    check_rs(rs_ai_allocate_slab(width * height * 4 + 4096, 256, 0, stream, &texture_slab), "allocate texture slab");
    RsAiTensorDesc preview_tensor{};
    preview_tensor.name = "preview_bgr";
    preview_tensor.group = "preview";
    preview_tensor.role = "ui_preview";
    preview_tensor.ndim = 3;
    preview_tensor.shape[0] = height;
    preview_tensor.shape[1] = width;
    preview_tensor.shape[2] = 3;
    preview_tensor.full_ndim = 3;
    preview_tensor.full_shape[0] = height;
    preview_tensor.full_shape[1] = width;
    preview_tensor.full_shape[2] = 3;
    preview_tensor.dtype = RS_AI_DTYPE_UINT8;
    preview_tensor.encoding = RS_AI_ENCODING_RAW_MASK_U8;
    preview_tensor.byte_size = host.size();
    RsAiDeviceFrame preview_frame{};
    preview_frame.desc.protocol_version = RS_AI_GPU_DIRECT_PROTOCOL_VERSION;
    preview_frame.desc.resource_mode = RS_AI_RESOURCE_ROI_LAYERED;
    preview_frame.desc.tensor_count = 1;
    preview_frame.desc.frame_id = 7;
    preview_frame.desc.source_h = height;
    preview_frame.desc.source_w = width;
    preview_frame.desc.network_h = width;
    preview_frame.desc.network_w = width;
    preview_frame.desc.payload_nbytes = host.size();
    preview_frame.desc.tensors = &preview_tensor;
    preview_frame.device_payload = static_cast<const std::uint8_t*>(d_image);
    preview_frame.device_payload_nbytes = host.size();
    preview_frame.cuda_stream = stream;
    RsAiDetection fallback_detection{};
    fallback_detection.frame_id = preview_frame.desc.frame_id;
    fallback_detection.x1 = 8.0f;
    fallback_detection.y1 = 8.0f;
    fallback_detection.x2 = 48.0f;
    fallback_detection.y2 = 36.0f;
    fallback_detection.confidence = 1.0f;
    RsAiDetectionList render_detections = detections;
    if (render_detections.count == 0 || render_detections.detections == nullptr) {
        render_detections.count = 1;
        render_detections.detections = &fallback_detection;
    }
    RsAiUiTexture texture{};
    std::cerr << "stage ui_render_begin render_detections=" << render_detections.count << "\n";
    check_rs(rs_ai_render_detections_to_texture(ui, &preview_frame, &render_detections, &texture_slab, &texture), "render texture");
    check_cuda(cudaStreamSynchronize(stream), "sync render texture");
    std::cerr << "stage ui_render_ok\n";
    if (texture.width != width || texture.height != height || texture.rgba.ptr == nullptr) {
        std::cerr << "ui texture metadata mismatch\n";
        return 7;
    }
    std::vector<std::uint8_t> rgba(static_cast<size_t>(texture.rgba.bytes), 0);
    check_cuda(cudaMemcpy(rgba.data(), texture.rgba.ptr, rgba.size(), cudaMemcpyDeviceToHost), "copy texture");
    bool found_box_pixel = false;
    for (size_t i = 0; i + 3 < rgba.size(); i += 4) {
        if (rgba[i + 0] == 0 && rgba[i + 1] == 255 && rgba[i + 2] == 64 && rgba[i + 3] == 255) {
            found_box_pixel = true;
            break;
        }
    }
    if (!found_box_pixel) {
        std::cerr << "ui texture did not contain detection overlay\n";
        return 8;
    }
    std::cout << "ai_c_abi_smoke_ok payload_bytes=" << device_frame.device_payload_nbytes
              << " detections=" << detections.count << "\n";
    std::cout << "ui_texture_smoke_ok rgba_bytes=" << texture.rgba.bytes
              << " size=" << texture.width << "x" << texture.height << "\n";
    rs_ai_release_detections(ai, &detections);
    rs_ai_release_slab(&texture_slab);
    rs_ai_destroy_ui_renderer(ui);
    rs_ai_release_slab(&ai_slab);
    rs_ai_destroy(ai);

    std::cout << "nvjpeg_smoke_ok jpeg_bytes=" << jpeg.bytes
              << " decoded=" << decoded.width << "x" << decoded.height << "\n";

    rs_ai_release_slab(&decode_slab);
    rs_ai_release_slab(&jpeg_slab);
    rs_ai_destroy_compressor(codec);
    cudaFree(d_image);
    cudaStreamDestroy(stream);
    std::cerr << "stage smoke_done\n";
    return 0;
}
