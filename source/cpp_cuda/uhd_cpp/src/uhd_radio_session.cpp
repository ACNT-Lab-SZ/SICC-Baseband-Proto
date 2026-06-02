#include "uhd_radio_session.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace usrp_link {
namespace {

const char* optional_cstr(const std::string& value)
{
    return value.empty() ? nullptr : value.c_str();
}

std::runtime_error radio_error(const std::string& prefix, const char* err)
{
    return std::runtime_error(prefix + ": " + (err != nullptr ? err : ""));
}

}  // namespace

RadioEndpoint::~RadioEndpoint()
{
    uhd_c_free(&handle_);
}

RadioEndpoint::RadioEndpoint(RadioEndpoint&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr))
{}

RadioEndpoint& RadioEndpoint::operator=(RadioEndpoint&& other) noexcept
{
    if (this != &other) {
        uhd_c_free(&handle_);
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

RadioEndpoint RadioEndpoint::open_tx(const RadioConfig& cfg)
{
    char err[2048] = {};
    uhd_c_radio_t* handle = nullptr;
    if (uhd_c_make_tx(
            &handle,
            cfg.args.c_str(),
            cfg.master_clock_rate,
            cfg.sample_rate,
            cfg.center_freq,
            cfg.gain,
            optional_cstr(cfg.clock_source),
            optional_cstr(cfg.time_source),
            optional_cstr(cfg.antenna),
            optional_cstr(cfg.subdev),
            optional_cstr(cfg.tx_cpu_format),
            cfg.channel,
            err,
            sizeof(err)) != 0) {
        throw radio_error("failed to create TX USRP", err);
    }
    return RadioEndpoint(handle);
}

RadioEndpoint RadioEndpoint::open_rx(const RadioConfig& cfg)
{
    char err[2048] = {};
    uhd_c_radio_t* handle = nullptr;
    if (uhd_c_make_rx(
            &handle,
            cfg.args.c_str(),
            cfg.master_clock_rate,
            cfg.sample_rate,
            cfg.center_freq,
            cfg.gain,
            optional_cstr(cfg.clock_source),
            optional_cstr(cfg.time_source),
            optional_cstr(cfg.antenna),
            optional_cstr(cfg.subdev),
            cfg.channel,
            err,
            sizeof(err)) != 0) {
        throw radio_error("failed to create RX USRP", err);
    }
    return RadioEndpoint(handle);
}

RadioEndpoint RadioEndpoint::open_trx(const DuplexRadioConfig& cfg)
{
    char err[2048] = {};
    uhd_c_radio_t* handle = nullptr;
    if (uhd_c_make_trx(
            &handle,
            cfg.args.c_str(),
            cfg.master_clock_rate,
            cfg.sample_rate,
            cfg.center_freq,
            cfg.tx_gain,
            cfg.rx_gain,
            optional_cstr(cfg.clock_source),
            optional_cstr(cfg.time_source),
            optional_cstr(cfg.tx_antenna),
            optional_cstr(cfg.rx_antenna),
            optional_cstr(cfg.tx_subdev),
            optional_cstr(cfg.rx_subdev),
            optional_cstr(cfg.tx_cpu_format),
            cfg.tx_channel,
            cfg.rx_channel,
            err,
            sizeof(err)) != 0) {
        throw radio_error("failed to create TRX USRP", err);
    }
    return RadioEndpoint(handle);
}

std::string RadioEndpoint::pp_string() const
{
    if (handle_ == nullptr) {
        return {};
    }
    std::vector<char> buffer(8192, '\0');
    if (uhd_c_pp_string(handle_, buffer.data(), buffer.size()) != 0) {
        return {};
    }
    return std::string(buffer.data());
}

size_t RadioEndpoint::tx_max_samps() const
{
    return handle_ != nullptr ? uhd_c_tx_max_samps(handle_) : 0;
}

size_t RadioEndpoint::rx_max_samps() const
{
    return handle_ != nullptr ? uhd_c_rx_max_samps(handle_) : 0;
}

void RadioEndpoint::send(
    const cf32* samples,
    size_t nsamps,
    bool start_of_burst,
    bool end_of_burst,
    double timeout,
    size_t& sent)
{
    char err[2048] = {};
    if (uhd_c_tx_send(
            handle_,
            samples,
            nsamps,
            start_of_burst ? 1 : 0,
            end_of_burst ? 1 : 0,
            timeout,
            &sent,
            err,
            sizeof(err)) != 0) {
        throw radio_error("TX send failed", err);
    }
}

void RadioEndpoint::send(
    const ci16* samples,
    size_t nsamps,
    bool start_of_burst,
    bool end_of_burst,
    double timeout,
    size_t& sent)
{
    char err[2048] = {};
    if (uhd_c_tx_send(
            handle_,
            samples,
            nsamps,
            start_of_burst ? 1 : 0,
            end_of_burst ? 1 : 0,
            timeout,
            &sent,
            err,
            sizeof(err)) != 0) {
        throw radio_error("TX send failed", err);
    }
}

void RadioEndpoint::send_end_of_burst(double timeout)
{
    char err[2048] = {};
    size_t sent = 0;
    if (uhd_c_tx_send(handle_, nullptr, 0, 0, 1, timeout, &sent, err, sizeof(err)) != 0) {
        throw radio_error("TX end-of-burst failed", err);
    }
}

void RadioEndpoint::start_rx()
{
    char err[2048] = {};
    if (uhd_c_rx_start(handle_, err, sizeof(err)) != 0) {
        throw radio_error("RX stream start failed", err);
    }
}

void RadioEndpoint::stop_rx()
{
    char err[2048] = {};
    if (uhd_c_rx_stop(handle_, err, sizeof(err)) != 0) {
        throw radio_error("RX stream stop failed", err);
    }
}

void RadioEndpoint::recv(cf32* samples, size_t max_samps, double timeout, size_t& received, int& status)
{
    char err[2048] = {};
    if (uhd_c_rx_recv(handle_, samples, max_samps, timeout, &received, &status, err, sizeof(err)) != 0) {
        throw radio_error("RX recv failed", err);
    }
}

TransceiverPair::TransceiverPair(RadioEndpoint tx_endpoint, RadioEndpoint rx_endpoint)
    : tx(std::move(tx_endpoint)),
      rx(std::move(rx_endpoint))
{}

TransceiverPair TransceiverPair::open(const TransceiverPairConfig& cfg)
{
    return TransceiverPair(RadioEndpoint::open_tx(cfg.tx), RadioEndpoint::open_rx(cfg.rx));
}

void set_realtime_priority()
{
    uhd_c_set_thread_priority();
}

}  // namespace usrp_link
