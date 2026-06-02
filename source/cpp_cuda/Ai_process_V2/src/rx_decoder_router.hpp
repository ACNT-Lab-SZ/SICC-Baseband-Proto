#pragma once

#include "fec_config.hpp"

#include <memory>
#include <string>

namespace fec {

enum class RxDecoderKind {
    GpuBp,
    GpuBpOsd,
    CpuMinSum,
    PolarScl,
};

struct RxDecoderRuntimeOptions {
    // "auto", "cpu", "cuda-bp", "cuda-bp-osd", or "polar-scl".
    std::string requested_decoder = "auto";
    int bp_max_iterations = 20;
    float bp_normalization = 0.80f;
    int cuda_min_batch = 30;
    int cuda_max_batch = 4096;
    int cuda_latency_us = 2000;
};

struct RxDecoderPlan {
    RxDecoderKind kind = RxDecoderKind::CpuMinSum;
    std::shared_ptr<const FecConfig> fec;
    std::string reason;
    bool requires_gpu = false;
};

class RxDecoderRouter {
public:
    static RxDecoderPlan make_plan(
        std::shared_ptr<const FecConfig> fec,
        const RxDecoderRuntimeOptions& options);

    static const char* to_string(RxDecoderKind kind);
};

}  // namespace fec
