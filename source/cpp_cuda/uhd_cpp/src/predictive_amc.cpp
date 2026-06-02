#include "predictive_amc.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace usrp_link {

namespace {

double clamp01(double x)
{
    return std::max(0.0, std::min(1.0, x));
}

StartupAmcDecision decision_for_tier(StartupAmcMode mode, int tier)
{
    StartupAmcDecision d;
    d.tier = std::max(0, std::min(4, tier));
    if (mode == StartupAmcMode::HighThroughput) {
        switch (d.tier) {
        case 0:
            d.modulation = "bpsk";
            d.fec_profile = "dvb-s2-short-r1/4";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_1_4.alist";
            d.fec_rate = 0.25;
            break;
        case 1:
            d.modulation = "qpsk";
            d.fec_profile = "dvb-s2-short-r1/2";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_1_2.alist";
            d.fec_rate = 0.5;
            break;
        case 2:
            d.modulation = "qpsk";
            d.fec_profile = "dvb-s2-short-r5/6";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_5_6.alist";
            d.fec_rate = 5.0 / 6.0;
            break;
        case 3:
            d.modulation = "16qam";
            d.fec_profile = "dvb-s2-short-r5/6";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_5_6.alist";
            d.fec_rate = 5.0 / 6.0;
            break;
        default:
            d.modulation = "64qam";
            d.fec_profile = "dvb-s2-short-r5/6";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_5_6.alist";
            d.fec_rate = 5.0 / 6.0;
            break;
        }
    } else {
        switch (d.tier) {
        case 0:
            d.modulation = "bpsk";
            d.fec_profile = "dvb-s2-short-r1/4";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_1_4.alist";
            d.fec_rate = 0.25;
            break;
        case 1:
            d.modulation = "qpsk";
            d.fec_profile = "dvb-s2-short-r1/4";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_1_4.alist";
            d.fec_rate = 0.25;
            break;
        case 2:
            d.modulation = "qpsk";
            d.fec_profile = "dvb-s2-short-r1/2";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_1_2.alist";
            d.fec_rate = 0.5;
            break;
        default:
            d.modulation = "qpsk";
            d.fec_profile = "dvb-s2-short-r5/6";
            d.fec_matrix_file = "DVB_S2_short_N16200_rate_5_6.alist";
            d.fec_rate = 5.0 / 6.0;
            break;
        }
    }
    return d;
}

} // namespace

StartupAmcDecision StartupAmcPolicy::select(const StartupAmcInput& input) const
{
    const bool throughput = input.mode == StartupAmcMode::HighThroughput;
    const double metric = input.link_source == StartupLinkSource::OfflineGpuSnr
        ? input.snr_db
        : input.tx_gain_db + input.rx_gain_db;

    int tier = 0;
    if (input.link_source == StartupLinkSource::OfflineGpuSnr) {
        if (throughput) {
            tier = metric < 1.5 ? 0 :
                   metric < 6.2 ? 1 :
                   metric < 13.2 ? 2 :
                   metric < 19.0 ? 3 : 4;
        } else {
            tier = metric < 0.0 ? 0 : metric < 5.0 ? 1 : metric < 10.0 ? 2 : 3;
        }
    } else if (throughput) {
        // The 16-QAM tier is enabled only at the measured 25 Msps RIO1
        // operating point (TX=23 dB, RX=24 dB) or above.
        tier = metric < 20.0 ? 0 : metric < 24.0 ? 1 : metric < 47.0 ? 2 : 3;
    } else {
        tier = metric < 20.0 ? 0 : metric < 24.0 ? 1 : 2;
    }

    std::string size_class = "stream-or-unknown";
    if (input.file_size_bytes > 0) {
        constexpr std::uint64_t kShortFileBytes = 4ull * 1024ull * 1024ull;
        constexpr std::uint64_t kLargeFileBytes = 64ull * 1024ull * 1024ull;
        if (input.file_size_bytes <= kShortFileBytes) {
            size_class = "short";
        } else if (input.file_size_bytes >= kLargeFileBytes) {
            size_class = "large";
        } else {
            size_class = "medium";
        }
    }

    auto d = decision_for_tier(input.mode, tier);
    if (size_class == "short") {
        d.fec_profile = "ccsds-ldpc-128-64";
        d.fec_matrix_file = "CCSDS_ldpc_n128_k64.alist";
        d.fec_rate = 0.5;
        if (!throughput) {
            d.modulation = metric < 4.0 ? "bpsk" : "qpsk";
        } else if (input.link_source == StartupLinkSource::OfflineGpuSnr) {
            d.modulation = metric < 4.0 ? "bpsk" :
                           metric < 11.1 ? "qpsk" :
                           metric < 16.8 ? "16qam" : "64qam";
        } else if (metric >= 47.0) {
            d.modulation = "64qam";
        }
    }
    d.selection_metric = metric;
    d.metric_name = input.link_source == StartupLinkSource::OfflineGpuSnr
        ? "snr_db"
        : "tx_gain_plus_rx_gain_db";
    d.size_class = size_class;
    std::ostringstream reason;
    reason << StartupAmcPolicy::to_string(input.mode) << " startup selection from "
           << d.metric_name << "=" << metric << " dB"
           << ", file=" << size_class;
    if (size_class == "short") {
        reason << "; CCSDS short-packet profile selected from measured threshold table";
    }
    if (!throughput) {
        reason << "; high-reliability mode limits aggressive MCS";
    }
    if (input.link_source == StartupLinkSource::UsrpGainBudget) {
        reason << "; USRP selection bounded by measured 25 Msps profiles";
    }
    d.reason = reason.str();
    return d;
}

const char* StartupAmcPolicy::to_string(StartupAmcMode mode)
{
    return mode == StartupAmcMode::HighThroughput ? "high-throughput" : "high-reliability";
}

PredictiveAmcPolicy::PredictiveAmcPolicy(PredictiveAmcConfig cfg)
    : cfg_(cfg),
      current_(table_decision(McsId::MCS1_QpskRepeat3, "initial conservative QPSK repeat=3"))
{
    cfg_.min_repeat = std::max(1, cfg_.min_repeat);
    cfg_.max_repeat = std::max(cfg_.min_repeat, cfg_.max_repeat);
    cfg_.hold_down_windows = std::max(0, cfg_.hold_down_windows);
    cfg_.upgrade_windows = std::max(1, cfg_.upgrade_windows);
}

ChannelQuality PredictiveAmcPolicy::evaluate_prediction(const ChannelPrediction& pred,
                                                        double snr_eff_db,
                                                        double fade_depth_db,
                                                        double evm) const
{
    const double age_penalty = std::max(0.0, pred.age_ms - cfg_.age_free_ms) * cfg_.age_penalty_db_per_ms;
    const double confidence = clamp01(pred.confidence);
    const double confidence_penalty = (1.0 - confidence) * cfg_.confidence_penalty_db;

    ChannelQuality q;
    q.snr_eff_db = snr_eff_db;
    q.fade_depth_db = fade_depth_db;
    q.evm = evm;
    q.snr_for_mcs_db = std::min({
        snr_eff_db,
        snr_eff_db + fade_depth_db * 0.35,
        snr_eff_db - age_penalty,
        snr_eff_db - confidence_penalty,
    });
    q.predicted_fade_ok = fade_depth_db > -8.0 && evm < 0.18 && confidence >= 0.75;
    return q;
}

AmcDecision PredictiveAmcPolicy::decide(const ChannelPrediction& pred,
                                        const ChannelQuality& quality,
                                        const RxFeedbackWindow* feedback)
{
    std::string reason = "prediction";
    McsId candidate = candidate_from_prediction(pred, quality);

    if (feedback != nullptr) {
        candidate = apply_feedback_guard(candidate, *feedback, reason);
    }

    const int current_rank = mcs_rank(current_.mcs);
    const int candidate_rank = mcs_rank(candidate);
    McsId selected = candidate;

    if (candidate_rank < current_rank) {
        hold_down_left_ = cfg_.hold_down_windows;
        upgrade_credit_ = 0;
        reason += "; immediate downgrade";
    } else if (candidate_rank > current_rank) {
        if (hold_down_left_ > 0) {
            --hold_down_left_;
            selected = current_.mcs;
            reason += "; hold-down blocks upgrade";
        } else {
            ++upgrade_credit_;
            if (upgrade_credit_ < cfg_.upgrade_windows) {
                selected = current_.mcs;
                reason += "; waiting for stable upgrade windows";
            } else {
                upgrade_credit_ = 0;
                reason += "; delayed upgrade accepted";
            }
        }
    } else {
        upgrade_credit_ = 0;
        if (hold_down_left_ > 0) {
            --hold_down_left_;
        }
    }

    current_ = table_decision(selected, reason);
    return current_;
}

AmcDecision PredictiveAmcPolicy::table_decision(McsId mcs, const std::string& reason) const
{
    AmcDecision d;
    d.mcs = mcs;
    d.reason = reason;
    d.fec_profile = "ccsds-ldpc-128-64";
    d.fec_rate = 0.5;

    switch (mcs) {
    case McsId::MCS0_ReliabilityBpskRepeat3:
        d.modulation = "bpsk";
        d.repeat_count = 3;
        break;
    case McsId::MCS1_QpskRepeat3:
        d.modulation = "qpsk";
        d.repeat_count = 3;
        break;
    case McsId::MCS2_QpskRepeat2:
        d.modulation = "qpsk";
        d.repeat_count = 2;
        break;
    case McsId::MCS3_QpskRepeat1:
        d.modulation = "qpsk";
        d.repeat_count = 1;
        break;
    case McsId::MCS4_16qamRepeat2:
        d.modulation = "16qam";
        d.repeat_count = 2;
        d.allow_runtime_switch = true;
        break;
    case McsId::MCS5_16qamRepeat1:
        d.modulation = "16qam";
        d.repeat_count = 1;
        d.allow_runtime_switch = true;
        break;
    }

    d.repeat_count = std::max(cfg_.min_repeat, std::min(cfg_.max_repeat, d.repeat_count));
    d.effective_rate = d.fec_rate / static_cast<double>(std::max(d.repeat_count, 1));
    return d;
}

McsId PredictiveAmcPolicy::candidate_from_prediction(const ChannelPrediction& pred, const ChannelQuality& q) const
{
    if (pred.paths.empty() || pred.confidence < 0.55f || q.snr_for_mcs_db < 6.0) {
        return McsId::MCS0_ReliabilityBpskRepeat3;
    }
    if (q.snr_for_mcs_db < 8.0) {
        return McsId::MCS1_QpskRepeat3;
    }
    if (q.snr_for_mcs_db < 10.0 || !q.predicted_fade_ok) {
        return McsId::MCS2_QpskRepeat2;
    }
    if (!cfg_.enable_16qam || q.snr_for_mcs_db < 15.0) {
        return McsId::MCS3_QpskRepeat1;
    }
    if (q.snr_for_mcs_db < 18.0 || q.evm >= 0.10) {
        return McsId::MCS4_16qamRepeat2;
    }
    return McsId::MCS5_16qamRepeat1;
}

McsId PredictiveAmcPolicy::apply_feedback_guard(McsId candidate,
                                                const RxFeedbackWindow& fb,
                                                std::string& reason)
{
    const bool severe = fb.fer >= 1.0e-2 || fb.sync_miss >= 2 || fb.overflow || fb.gpu_queue_drop;
    const bool target_miss = fb.frames > 0 && fb.fer >= cfg_.target_fer;

    if (severe) {
        reason += "; severe RX feedback";
        return std::min(candidate, McsId::MCS1_QpskRepeat3, [](McsId a, McsId b) {
            return mcs_rank(a) < mcs_rank(b);
        });
    }
    if (target_miss) {
        reason += "; FER guard";
        return rank_to_mcs(std::max(0, mcs_rank(candidate) - 1));
    }
    if (fb.frames > 0 && fb.fer > cfg_.upgrade_fer) {
        reason += "; upgrade disabled by feedback margin";
        return std::min(candidate, current_.mcs, [](McsId a, McsId b) {
            return mcs_rank(a) < mcs_rank(b);
        });
    }
    reason += "; feedback clean";
    return candidate;
}

int PredictiveAmcPolicy::mcs_rank(McsId mcs)
{
    return static_cast<int>(mcs);
}

McsId PredictiveAmcPolicy::rank_to_mcs(int rank)
{
    rank = std::max(0, std::min(5, rank));
    return static_cast<McsId>(rank);
}

} // namespace usrp_link
