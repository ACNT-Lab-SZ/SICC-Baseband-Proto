#include "rx_decoder_router.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <type_traits>

namespace fec {
namespace {

static std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static void validate_gpu_bp_input(const FecConfig& cfg)
{
    std::visit(
        [](const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, DvbS2LdpcParams>) {
                if (p.h.row_ptr.empty() || p.h.col_ind.empty() || p.h.val.empty()) {
                    throw std::runtime_error("GPU BP routing requires non-empty CSR matrix");
                }
            } else if constexpr (std::is_same_v<T, NrLdpcParams>) {
                if (p.lifting_factor_zc <= 0) {
                    throw std::runtime_error("GPU BP routing requires a valid NR lifting factor");
                }
            } else if constexpr (std::is_same_v<T, PolarCodeParams>) {
                // The strict routing rule is based on block length. A long
                // Polar configuration reaches GPU BP only if an outer caller
                // intentionally maps it to a BP-compatible graph.
            }
        },
        cfg.params);
}

}  // namespace

RxDecoderPlan RxDecoderRouter::make_plan(
    std::shared_ptr<const FecConfig> cfg,
    const RxDecoderRuntimeOptions& options)
{
    if (!cfg) {
        throw std::runtime_error("RX decoder routing received a null FEC config");
    }
    if (cfg->block_length() <= 0 || cfg->information_length() <= 0) {
        throw std::runtime_error("RX decoder routing received invalid FEC dimensions");
    }

    const std::string requested = lower(options.requested_decoder);

    // Strict rule: any N > 1000 is forced to GPU BP, regardless of the
    // requested decoder string.
    if (cfg->block_length() > 1000) {
        validate_gpu_bp_input(*cfg);
        return RxDecoderPlan{
            RxDecoderKind::GpuBp,
            std::move(cfg),
            "block_length > 1000: forced GPU BP decoder",
            true};
    }

    if (requested == "cuda-bp-osd" || requested == "gpu-bp-osd") {
        return RxDecoderPlan{
            RxDecoderKind::GpuBpOsd,
            std::move(cfg),
            "explicit short-code CUDA BP-OSD request",
            true};
    }

    if (requested == "cuda-osd" || requested == "gpu-osd" || requested == "osd-only") {
        return RxDecoderPlan{
            RxDecoderKind::GpuBpOsd,
            std::move(cfg),
            "explicit short-code CUDA OSD-only request",
            true};
    }

    if (requested == "polar-scl" || requested == "scl") {
        if (!std::holds_alternative<PolarCodeParams>(cfg->params)) {
            throw std::runtime_error("polar-scl requested for a non-Polar FEC config");
        }
        return RxDecoderPlan{
            RxDecoderKind::PolarScl,
            std::move(cfg),
            "explicit short Polar SCL request",
            false};
    }

    if (requested == "cuda-bp" || requested == "gpu-bp") {
        validate_gpu_bp_input(*cfg);
        return RxDecoderPlan{
            RxDecoderKind::GpuBp,
            std::move(cfg),
            "explicit GPU BP request",
            true};
    }

    if (std::holds_alternative<PolarCodeParams>(cfg->params)) {
        return RxDecoderPlan{
            RxDecoderKind::PolarScl,
            std::move(cfg),
            "auto: short Polar code routes to SCL",
            false};
    }

    return RxDecoderPlan{
        RxDecoderKind::CpuMinSum,
        std::move(cfg),
        "auto: short LDPC code routes to CPU min-sum",
        false};
}

const char* RxDecoderRouter::to_string(RxDecoderKind kind)
{
    switch (kind) {
    case RxDecoderKind::GpuBp:
        return "gpu-bp";
    case RxDecoderKind::GpuBpOsd:
        return "gpu-bp-osd";
    case RxDecoderKind::CpuMinSum:
        return "cpu-min-sum";
    case RxDecoderKind::PolarScl:
        return "polar-scl";
    }
    return "unknown";
}

}  // namespace fec
