#pragma once

#include "uhd_c_radio.h"

#include <complex>
#include <cstddef>
#include <string>

namespace usrp_link {

using cf32 = std::complex<float>;

struct RadioConfig {
    std::string args;
    double master_clock_rate = 200e6;
    double sample_rate = 2e6;
    double center_freq = 2.45e9;
    double gain = 0.0;
    std::string clock_source = "internal";
    std::string time_source = "internal";
    std::string antenna;
    std::string subdev;
    size_t channel = 0;
};

struct DuplexRadioConfig {
    std::string args;
    double master_clock_rate = 200e6;
    double sample_rate = 2e6;
    double center_freq = 2.45e9;
    double tx_gain = 0.0;
    double rx_gain = 0.0;
    std::string clock_source = "internal";
    std::string time_source = "internal";
    std::string tx_antenna;
    std::string rx_antenna;
    std::string tx_subdev;
    std::string rx_subdev;
    size_t tx_channel = 0;
    size_t rx_channel = 0;
};

class RadioEndpoint {
public:
    RadioEndpoint() = default;
    ~RadioEndpoint();

    RadioEndpoint(const RadioEndpoint&) = delete;
    RadioEndpoint& operator=(const RadioEndpoint&) = delete;
    RadioEndpoint(RadioEndpoint&& other) noexcept;
    RadioEndpoint& operator=(RadioEndpoint&& other) noexcept;

    static RadioEndpoint open_tx(const RadioConfig& cfg);
    static RadioEndpoint open_rx(const RadioConfig& cfg);
    static RadioEndpoint open_trx(const DuplexRadioConfig& cfg);

    bool valid() const noexcept { return handle_ != nullptr; }
    std::string pp_string() const;

    size_t tx_max_samps() const;
    size_t rx_max_samps() const;

    void send(const cf32* samples, size_t nsamps, bool start_of_burst, bool end_of_burst, double timeout, size_t& sent);
    void send_end_of_burst(double timeout);

    void start_rx();
    void stop_rx();
    void recv(cf32* samples, size_t max_samps, double timeout, size_t& received, int& status);

private:
    explicit RadioEndpoint(uhd_c_radio_t* handle) : handle_(handle) {}

    uhd_c_radio_t* handle_ = nullptr;
};

struct TransceiverPairConfig {
    RadioConfig tx;
    RadioConfig rx;
};

class TransceiverPair {
public:
    static TransceiverPair open(const TransceiverPairConfig& cfg);

    RadioEndpoint tx;
    RadioEndpoint rx;

private:
    TransceiverPair(RadioEndpoint tx_endpoint, RadioEndpoint rx_endpoint);
};

void set_realtime_priority();

}  // namespace usrp_link
