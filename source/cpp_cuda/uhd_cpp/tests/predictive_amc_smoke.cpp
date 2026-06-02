#include "predictive_amc.hpp"

#include <iostream>
#include <stdexcept>

using namespace usrp_link;

namespace {

ChannelPrediction make_prediction(float confidence = 0.9f, double age_ms = 1.0)
{
    ChannelPrediction pred;
    pred.prediction_id = 1;
    pred.target_phy_frame_id = 100;
    pred.age_ms = age_ms;
    pred.confidence = confidence;
    pred.paths.push_back(PredictedPath{1.0f, 0.0f, 0.0f, 0.0f});
    pred.paths.push_back(PredictedPath{0.4f, 0.2f, 1.5f, 0.2f});
    return pred;
}

void require(bool ok, const char* msg)
{
    if (!ok) {
        throw std::runtime_error(msg);
    }
}

} // namespace

int main()
{
    StartupAmcPolicy startup_policy;
    StartupAmcInput offline;
    offline.mode = StartupAmcMode::HighThroughput;
    offline.link_source = StartupLinkSource::OfflineGpuSnr;
    offline.snr_db = 20.0;
    offline.file_size_bytes = 100ull * 1024ull * 1024ull;
    auto startup = startup_policy.select(offline);
    require(startup.modulation == "64qam", "high-throughput offline SNR should allow 64QAM");
    require(startup.fec_matrix_file == "DVB_S2_short_N16200_rate_5_6.alist",
            "high-throughput startup code-rate mismatch");

    StartupAmcInput reliable_usrp;
    reliable_usrp.mode = StartupAmcMode::HighReliability;
    reliable_usrp.link_source = StartupLinkSource::UsrpGainBudget;
    reliable_usrp.tx_gain_db = 23.0;
    reliable_usrp.rx_gain_db = 24.0;
    reliable_usrp.file_size_bytes = 100ull * 1024ull * 1024ull;
    startup = startup_policy.select(reliable_usrp);
    require(startup.modulation == "qpsk", "reliability mode should retain QPSK at calibrated USRP gains");
    require(startup.fec_matrix_file == "DVB_S2_short_N16200_rate_1_2.alist",
            "reliability startup code-rate mismatch");

    offline.file_size_bytes = 1024;
    startup = startup_policy.select(offline);
    require(startup.modulation == "64qam", "calibrated short-file tier should allow 64QAM at high SNR");
    require(startup.fec_matrix_file == "CCSDS_ldpc_n128_k64.alist",
            "short-file startup should use CCSDS n128/k64");

    StartupAmcInput throughput_usrp;
    throughput_usrp.mode = StartupAmcMode::HighThroughput;
    throughput_usrp.link_source = StartupLinkSource::UsrpGainBudget;
    throughput_usrp.tx_gain_db = 23.0;
    throughput_usrp.rx_gain_db = 24.0;
    throughput_usrp.file_size_bytes = 100ull * 1024ull * 1024ull;
    startup = startup_policy.select(throughput_usrp);
    require(startup.modulation == "16qam", "calibrated USRP stream tier should allow 16QAM");
    require(startup.fec_matrix_file == "DVB_S2_short_N16200_rate_5_6.alist",
            "calibrated USRP stream high-throughput code-rate mismatch");

    throughput_usrp.file_size_bytes = 1024;
    startup = startup_policy.select(throughput_usrp);
    require(startup.modulation == "64qam", "calibrated USRP short-packet tier should allow 64QAM");
    require(startup.fec_matrix_file == "CCSDS_ldpc_n128_k64.alist",
            "calibrated USRP short-packet matrix mismatch");

    PredictiveAmcConfig cfg;
    cfg.enable_16qam = true;
    PredictiveAmcPolicy policy(cfg);

    auto pred = make_prediction();
    auto q = policy.evaluate_prediction(pred, 12.0, -2.0, 0.05);

    AmcDecision d;
    for (int i = 0; i < cfg.upgrade_windows; ++i) {
        d = policy.decide(pred, q, nullptr);
    }
    require(d.mcs == McsId::MCS3_QpskRepeat1, "high-quality prediction should reach QPSK repeat=1");
    require(d.repeat_count == 1, "MCS3 repeat count mismatch");

    RxFeedbackWindow bad;
    bad.frames = 1000;
    bad.ok_frames = 998;
    bad.fer = 2.0e-3;
    d = policy.decide(pred, q, &bad);
    require(static_cast<int>(d.mcs) <= static_cast<int>(McsId::MCS2_QpskRepeat2),
            "FER guard should downgrade at least one MCS");

    auto weak_pred = make_prediction(0.4f, 1.0);
    auto weak_q = policy.evaluate_prediction(weak_pred, 14.0, -1.0, 0.03);
    d = policy.decide(weak_pred, weak_q, nullptr);
    require(d.mcs == McsId::MCS0_ReliabilityBpskRepeat3,
            "low-confidence prediction should select reliability MCS0");

    std::cout << "predictive_amc_smoke: ok\n";
    std::cout << "last modulation=" << d.modulation
              << " repeat=" << d.repeat_count
              << " effective_rate=" << d.effective_rate
              << " reason=\"" << d.reason << "\"\n";
    return 0;
}
