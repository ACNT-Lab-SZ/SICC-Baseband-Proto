#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace usrp_link {

struct PredictedPath {
    float h_re = 1.0f;
    float h_im = 0.0f;
    float doppler_hz = 0.0f;
    float delay_samples = 0.0f;
};

struct ChannelPrediction {
    std::uint64_t prediction_id = 0;
    std::uint64_t target_phy_frame_id = 0;
    double age_ms = 0.0;
    float confidence = 1.0f;
    std::vector<PredictedPath> paths;
};

struct ChannelQuality {
    double snr_eff_db = 0.0;
    double fade_depth_db = 0.0;
    double evm = 0.0;
    double snr_for_mcs_db = 0.0;
    bool predicted_fade_ok = false;
};

struct RxFeedbackWindow {
    int frames = 0;
    int ok_frames = 0;
    double fer = 0.0;
    double ber = 0.0;
    double avg_snr_db = 0.0;
    int sync_miss = 0;
    bool overflow = false;
    bool gpu_queue_drop = false;
};

enum class McsId {
    MCS0_ReliabilityBpskRepeat3 = 0,
    MCS1_QpskRepeat3 = 1,
    MCS2_QpskRepeat2 = 2,
    MCS3_QpskRepeat1 = 3,
    MCS4_16qamRepeat2 = 4,
    MCS5_16qamRepeat1 = 5,
};

struct AmcDecision {
    McsId mcs = McsId::MCS1_QpskRepeat3;
    std::string modulation = "qpsk";
    std::string fec_profile = "ccsds-ldpc-128-64";
    double fec_rate = 0.5;
    int repeat_count = 3;
    double effective_rate = 1.0 / 6.0;
    bool allow_runtime_switch = false;
    std::string reason;
};

struct PredictiveAmcConfig {
    double target_fer = 1.0e-3;
    double upgrade_fer = 5.0e-4;
    int min_repeat = 1;
    int max_repeat = 3;
    int hold_down_windows = 2;
    int upgrade_windows = 3;
    double age_free_ms = 5.0;
    double age_penalty_db_per_ms = 0.15;
    double confidence_penalty_db = 4.0;
    bool enable_16qam = false;
};

enum class StartupAmcMode {
    HighThroughput,
    HighReliability,
};

enum class StartupLinkSource {
    OfflineGpuSnr,
    UsrpGainBudget,
};

struct StartupAmcInput {
    StartupAmcMode mode = StartupAmcMode::HighThroughput;
    StartupLinkSource link_source = StartupLinkSource::OfflineGpuSnr;
    double snr_db = 0.0;
    double tx_gain_db = 0.0;
    double rx_gain_db = 0.0;
    std::uint64_t file_size_bytes = 0;
};

struct StartupAmcDecision {
    std::string modulation = "qpsk";
    std::string fec_profile = "dvb-s2-short-r1/2";
    std::string fec_matrix_file = "DVB_S2_short_N16200_rate_1_2.alist";
    double fec_rate = 0.5;
    int tier = 0;
    double selection_metric = 0.0;
    std::string metric_name;
    std::string size_class;
    std::string reason;
};

class StartupAmcPolicy {
public:
    StartupAmcDecision select(const StartupAmcInput& input) const;
    static const char* to_string(StartupAmcMode mode);
};

class PredictiveAmcPolicy {
public:
    explicit PredictiveAmcPolicy(PredictiveAmcConfig cfg = {});

    ChannelQuality evaluate_prediction(const ChannelPrediction& pred,
                                       double snr_eff_db,
                                       double fade_depth_db,
                                       double evm) const;

    AmcDecision decide(const ChannelPrediction& pred,
                       const ChannelQuality& quality,
                       const RxFeedbackWindow* feedback);

    AmcDecision current_decision() const { return current_; }

private:
    AmcDecision table_decision(McsId mcs, const std::string& reason) const;
    McsId candidate_from_prediction(const ChannelPrediction& pred, const ChannelQuality& q) const;
    McsId apply_feedback_guard(McsId candidate, const RxFeedbackWindow& fb, std::string& reason);
    static int mcs_rank(McsId mcs);
    static McsId rank_to_mcs(int rank);

    PredictiveAmcConfig cfg_;
    AmcDecision current_;
    int hold_down_left_ = 0;
    int upgrade_credit_ = 0;
};

} // namespace usrp_link
