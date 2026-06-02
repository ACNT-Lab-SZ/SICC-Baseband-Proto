#include "uhd_c_radio.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <uhd.h>

struct uhd_c_radio {
    uhd_usrp_handle usrp;
    uhd_tx_streamer_handle tx_stream;
    uhd_rx_streamer_handle rx_stream;
    uhd_tx_metadata_handle tx_md_start;
    uhd_tx_metadata_handle tx_md_cont;
    uhd_tx_metadata_handle tx_md_end;
    uhd_rx_metadata_handle rx_md;
};

static void set_err(char* err, size_t err_len, const char* msg)
{
    if (err == NULL || err_len == 0) {
        return;
    }
    if (msg == NULL) {
        msg = "unknown UHD C API error";
    }
    snprintf(err, err_len, "%s", msg);
}

static int check_global(uhd_error e, char* err, size_t err_len)
{
    if (e == UHD_ERROR_NONE) {
        return 0;
    }
    char buf[2048];
    buf[0] = '\0';
    uhd_get_last_error(buf, sizeof(buf));
    set_err(err, err_len, buf[0] ? buf : "UHD C API call failed");
    return -1;
}

static int check_usrp(uhd_error e, uhd_usrp_handle usrp, char* err, size_t err_len)
{
    if (e == UHD_ERROR_NONE) {
        return 0;
    }
    char buf[2048];
    buf[0] = '\0';
    if (usrp != NULL) {
        uhd_usrp_last_error(usrp, buf, sizeof(buf));
    }
    if (!buf[0]) {
        uhd_get_last_error(buf, sizeof(buf));
    }
    set_err(err, err_len, buf[0] ? buf : "USRP API call failed");
    return -1;
}

static int check_tx(uhd_error e, uhd_tx_streamer_handle tx, char* err, size_t err_len)
{
    if (e == UHD_ERROR_NONE) {
        return 0;
    }
    char buf[2048];
    buf[0] = '\0';
    if (tx != NULL) {
        uhd_tx_streamer_last_error(tx, buf, sizeof(buf));
    }
    if (!buf[0]) {
        uhd_get_last_error(buf, sizeof(buf));
    }
    set_err(err, err_len, buf[0] ? buf : "TX streamer API call failed");
    return -1;
}

static int check_rx(uhd_error e, uhd_rx_streamer_handle rx, char* err, size_t err_len)
{
    if (e == UHD_ERROR_NONE) {
        return 0;
    }
    char buf[2048];
    buf[0] = '\0';
    if (rx != NULL) {
        uhd_rx_streamer_last_error(rx, buf, sizeof(buf));
    }
    if (!buf[0]) {
        uhd_get_last_error(buf, sizeof(buf));
    }
    set_err(err, err_len, buf[0] ? buf : "RX streamer API call failed");
    return -1;
}

static uhd_tune_request_t tune_request(double freq)
{
    uhd_tune_request_t tr;
    memset(&tr, 0, sizeof(tr));
    tr.target_freq = freq;
    tr.rf_freq_policy = UHD_TUNE_REQUEST_POLICY_AUTO;
    tr.dsp_freq_policy = UHD_TUNE_REQUEST_POLICY_AUTO;
    tr.args = "";
    return tr;
}

static int apply_common(
    uhd_c_radio_t* r,
    const char* args,
    double master_clock_rate,
    const char* clock_source,
    const char* time_source,
    char* err,
    size_t err_len)
{
    if (check_global(uhd_usrp_make(&r->usrp, args), err, err_len) != 0) {
        return -1;
    }
    if (clock_source != NULL && clock_source[0] != '\0') {
        if (check_usrp(uhd_usrp_set_clock_source(r->usrp, clock_source, 0), r->usrp, err, err_len) != 0) {
            return -1;
        }
    }
    if (time_source != NULL && time_source[0] != '\0') {
        if (check_usrp(uhd_usrp_set_time_source(r->usrp, time_source, 0), r->usrp, err, err_len) != 0) {
            return -1;
        }
    }
    if (master_clock_rate > 0.0) {
        if (check_usrp(uhd_usrp_set_master_clock_rate(r->usrp, master_clock_rate, 0), r->usrp, err, err_len) != 0) {
            return -1;
        }
    }
    return 0;
}

static int apply_subdev(uhd_c_radio_t* r, const char* subdev, int is_tx, char* err, size_t err_len)
{
    if (subdev == NULL || subdev[0] == '\0') {
        return 0;
    }

    uhd_subdev_spec_handle spec = NULL;
    if (check_global(uhd_subdev_spec_make(&spec, subdev), err, err_len) != 0) {
        return -1;
    }

    uhd_error e = is_tx
        ? uhd_usrp_set_tx_subdev_spec(r->usrp, spec, 0)
        : uhd_usrp_set_rx_subdev_spec(r->usrp, spec, 0);
    uhd_subdev_spec_free(&spec);

    return check_usrp(e, r->usrp, err, err_len);
}

int uhd_c_make_tx(
    uhd_c_radio_t** out,
    const char* args,
    double master_clock_rate,
    double rate,
    double freq,
    double gain,
    const char* clock_source,
    const char* time_source,
    const char* antenna,
    const char* subdev,
    const char* tx_cpu_format,
    size_t channel,
    char* err,
    size_t err_len)
{
    *out = NULL;
    uhd_c_radio_t* r = (uhd_c_radio_t*)calloc(1, sizeof(*r));
    if (r == NULL) {
        set_err(err, err_len, "out of memory");
        return -1;
    }

    if (apply_common(r, args, master_clock_rate, clock_source, time_source, err, err_len) != 0 ||
        apply_subdev(r, subdev, 1, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_tx_rate(r->usrp, rate, channel), r->usrp, err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    uhd_tune_request_t tr = tune_request(freq);
    uhd_tune_result_t tune_result;
    memset(&tune_result, 0, sizeof(tune_result));
    if (check_usrp(uhd_usrp_set_tx_freq(r->usrp, &tr, channel, &tune_result), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_tx_gain(r->usrp, gain, channel, ""), r->usrp, err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    if (antenna != NULL && antenna[0] != '\0') {
        if (check_usrp(uhd_usrp_set_tx_antenna(r->usrp, antenna, channel), r->usrp, err, err_len) != 0) {
            uhd_c_free(&r);
            return -1;
        }
    }

    if (check_global(uhd_tx_streamer_make(&r->tx_stream), err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    size_t channels[1] = {channel};
    uhd_stream_args_t sa;
    memset(&sa, 0, sizeof(sa));
        sa.cpu_format = (char*)((tx_cpu_format != NULL && tx_cpu_format[0] != '\0') ? tx_cpu_format : "fc32");
    sa.otw_format = "sc16";
    sa.args = "";
    sa.channel_list = channels;
    sa.n_channels = 1;
    if (check_usrp(uhd_usrp_get_tx_stream(r->usrp, &sa, r->tx_stream), r->usrp, err, err_len) != 0 ||
        check_global(uhd_tx_metadata_make(&r->tx_md_start, false, 0, 0.0, true, false), err, err_len) != 0 ||
        check_global(uhd_tx_metadata_make(&r->tx_md_cont, false, 0, 0.0, false, false), err, err_len) != 0 ||
        check_global(uhd_tx_metadata_make(&r->tx_md_end, false, 0, 0.0, false, true), err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    *out = r;
    return 0;
}

int uhd_c_make_rx(
    uhd_c_radio_t** out,
    const char* args,
    double master_clock_rate,
    double rate,
    double freq,
    double gain,
    const char* clock_source,
    const char* time_source,
    const char* antenna,
    const char* subdev,
    size_t channel,
    char* err,
    size_t err_len)
{
    *out = NULL;
    uhd_c_radio_t* r = (uhd_c_radio_t*)calloc(1, sizeof(*r));
    if (r == NULL) {
        set_err(err, err_len, "out of memory");
        return -1;
    }

    if (apply_common(r, args, master_clock_rate, clock_source, time_source, err, err_len) != 0 ||
        apply_subdev(r, subdev, 0, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_rx_rate(r->usrp, rate, channel), r->usrp, err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    uhd_tune_request_t tr = tune_request(freq);
    uhd_tune_result_t tune_result;
    memset(&tune_result, 0, sizeof(tune_result));
    if (check_usrp(uhd_usrp_set_rx_freq(r->usrp, &tr, channel, &tune_result), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_rx_gain(r->usrp, gain, channel, ""), r->usrp, err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    if (antenna != NULL && antenna[0] != '\0') {
        if (check_usrp(uhd_usrp_set_rx_antenna(r->usrp, antenna, channel), r->usrp, err, err_len) != 0) {
            uhd_c_free(&r);
            return -1;
        }
    }

    if (check_global(uhd_rx_streamer_make(&r->rx_stream), err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    size_t channels[1] = {channel};
    uhd_stream_args_t sa;
    memset(&sa, 0, sizeof(sa));
    sa.cpu_format = "fc32";
    sa.otw_format = "sc16";
    sa.args = "";
    sa.channel_list = channels;
    sa.n_channels = 1;
    if (check_usrp(uhd_usrp_get_rx_stream(r->usrp, &sa, r->rx_stream), r->usrp, err, err_len) != 0 ||
        check_global(uhd_rx_metadata_make(&r->rx_md), err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    *out = r;
    return 0;
}

int uhd_c_make_trx(
    uhd_c_radio_t** out,
    const char* args,
    double master_clock_rate,
    double rate,
    double freq,
    double tx_gain,
    double rx_gain,
    const char* clock_source,
    const char* time_source,
    const char* tx_antenna,
    const char* rx_antenna,
    const char* tx_subdev,
    const char* rx_subdev,
    const char* tx_cpu_format,
    size_t tx_channel,
    size_t rx_channel,
    char* err,
    size_t err_len)
{
    *out = NULL;
    uhd_c_radio_t* r = (uhd_c_radio_t*)calloc(1, sizeof(*r));
    if (r == NULL) {
        set_err(err, err_len, "out of memory");
        return -1;
    }

    if (apply_common(r, args, master_clock_rate, clock_source, time_source, err, err_len) != 0 ||
        apply_subdev(r, tx_subdev, 1, err, err_len) != 0 ||
        apply_subdev(r, rx_subdev, 0, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_tx_rate(r->usrp, rate, tx_channel), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_rx_rate(r->usrp, rate, rx_channel), r->usrp, err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    uhd_tune_request_t tx_tr = tune_request(freq);
    uhd_tune_request_t rx_tr = tune_request(freq);
    uhd_tune_result_t tx_tune_result;
    uhd_tune_result_t rx_tune_result;
    memset(&tx_tune_result, 0, sizeof(tx_tune_result));
    memset(&rx_tune_result, 0, sizeof(rx_tune_result));
    if (check_usrp(uhd_usrp_set_tx_freq(r->usrp, &tx_tr, tx_channel, &tx_tune_result), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_rx_freq(r->usrp, &rx_tr, rx_channel, &rx_tune_result), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_tx_gain(r->usrp, tx_gain, tx_channel, ""), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_set_rx_gain(r->usrp, rx_gain, rx_channel, ""), r->usrp, err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    if (tx_antenna != NULL && tx_antenna[0] != '\0') {
        if (check_usrp(uhd_usrp_set_tx_antenna(r->usrp, tx_antenna, tx_channel), r->usrp, err, err_len) != 0) {
            uhd_c_free(&r);
            return -1;
        }
    }
    if (rx_antenna != NULL && rx_antenna[0] != '\0') {
        if (check_usrp(uhd_usrp_set_rx_antenna(r->usrp, rx_antenna, rx_channel), r->usrp, err, err_len) != 0) {
            uhd_c_free(&r);
            return -1;
        }
    }

    if (check_global(uhd_tx_streamer_make(&r->tx_stream), err, err_len) != 0 ||
        check_global(uhd_rx_streamer_make(&r->rx_stream), err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    size_t tx_channels[1] = {tx_channel};
    uhd_stream_args_t tx_sa;
    memset(&tx_sa, 0, sizeof(tx_sa));
        tx_sa.cpu_format = (char*)((tx_cpu_format != NULL && tx_cpu_format[0] != '\0') ? tx_cpu_format : "fc32");
    tx_sa.otw_format = "sc16";
    tx_sa.args = "";
    tx_sa.channel_list = tx_channels;
    tx_sa.n_channels = 1;

    size_t rx_channels[1] = {rx_channel};
    uhd_stream_args_t rx_sa;
    memset(&rx_sa, 0, sizeof(rx_sa));
    rx_sa.cpu_format = "fc32";
    rx_sa.otw_format = "sc16";
    rx_sa.args = "";
    rx_sa.channel_list = rx_channels;
    rx_sa.n_channels = 1;

    if (check_usrp(uhd_usrp_get_tx_stream(r->usrp, &tx_sa, r->tx_stream), r->usrp, err, err_len) != 0 ||
        check_usrp(uhd_usrp_get_rx_stream(r->usrp, &rx_sa, r->rx_stream), r->usrp, err, err_len) != 0 ||
        check_global(uhd_tx_metadata_make(&r->tx_md_start, false, 0, 0.0, true, false), err, err_len) != 0 ||
        check_global(uhd_tx_metadata_make(&r->tx_md_cont, false, 0, 0.0, false, false), err, err_len) != 0 ||
        check_global(uhd_tx_metadata_make(&r->tx_md_end, false, 0, 0.0, false, true), err, err_len) != 0 ||
        check_global(uhd_rx_metadata_make(&r->rx_md), err, err_len) != 0) {
        uhd_c_free(&r);
        return -1;
    }

    *out = r;
    return 0;
}

void uhd_c_free(uhd_c_radio_t** radio)
{
    if (radio == NULL || *radio == NULL) {
        return;
    }
    uhd_c_radio_t* r = *radio;
    if (r->rx_md != NULL) {
        uhd_rx_metadata_free(&r->rx_md);
    }
    if (r->tx_md_start != NULL) {
        uhd_tx_metadata_free(&r->tx_md_start);
    }
    if (r->tx_md_cont != NULL) {
        uhd_tx_metadata_free(&r->tx_md_cont);
    }
    if (r->tx_md_end != NULL) {
        uhd_tx_metadata_free(&r->tx_md_end);
    }
    if (r->rx_stream != NULL) {
        uhd_rx_streamer_free(&r->rx_stream);
    }
    if (r->tx_stream != NULL) {
        uhd_tx_streamer_free(&r->tx_stream);
    }
    if (r->usrp != NULL) {
        uhd_usrp_free(&r->usrp);
    }
    free(r);
    *radio = NULL;
}

size_t uhd_c_tx_max_samps(uhd_c_radio_t* radio)
{
    size_t n = 0;
    if (radio != NULL && radio->tx_stream != NULL) {
        uhd_tx_streamer_max_num_samps(radio->tx_stream, &n);
    }
    return n;
}

size_t uhd_c_rx_max_samps(uhd_c_radio_t* radio)
{
    size_t n = 0;
    if (radio != NULL && radio->rx_stream != NULL) {
        uhd_rx_streamer_max_num_samps(radio->rx_stream, &n);
    }
    return n;
}

int uhd_c_tx_send(
    uhd_c_radio_t* radio,
    const void* samples,
    size_t nsamps,
    int start_of_burst,
    int end_of_burst,
    double timeout,
    size_t* sent,
    char* err,
    size_t err_len)
{
    if (sent != NULL) {
        *sent = 0;
    }
    if (radio == NULL || radio->tx_stream == NULL) {
        set_err(err, err_len, "TX radio is not initialized");
        return -1;
    }

    const void* buffs[1] = {samples};
    uhd_tx_metadata_handle* md = &radio->tx_md_cont;
    if (end_of_burst) {
        md = &radio->tx_md_end;
    } else if (start_of_burst) {
        md = &radio->tx_md_start;
    }

    size_t local_sent = 0;
    if (check_tx(uhd_tx_streamer_send(radio->tx_stream, buffs, nsamps, md, timeout, &local_sent), radio->tx_stream, err, err_len) != 0) {
        return -1;
    }
    if (sent != NULL) {
        *sent = local_sent;
    }
    return 0;
}

int uhd_c_rx_start(uhd_c_radio_t* radio, char* err, size_t err_len)
{
    if (radio == NULL || radio->rx_stream == NULL) {
        set_err(err, err_len, "RX radio is not initialized");
        return -1;
    }
    uhd_stream_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.stream_mode = UHD_STREAM_MODE_START_CONTINUOUS;
    cmd.stream_now = true;
    return check_rx(uhd_rx_streamer_issue_stream_cmd(radio->rx_stream, &cmd), radio->rx_stream, err, err_len);
}

int uhd_c_rx_stop(uhd_c_radio_t* radio, char* err, size_t err_len)
{
    if (radio == NULL || radio->rx_stream == NULL) {
        return 0;
    }
    uhd_stream_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.stream_mode = UHD_STREAM_MODE_STOP_CONTINUOUS;
    cmd.stream_now = true;
    return check_rx(uhd_rx_streamer_issue_stream_cmd(radio->rx_stream, &cmd), radio->rx_stream, err, err_len);
}

int uhd_c_rx_recv(
    uhd_c_radio_t* radio,
    void* samples,
    size_t max_samps,
    double timeout,
    size_t* received,
    int* status,
    char* err,
    size_t err_len)
{
    if (received != NULL) {
        *received = 0;
    }
    if (status != NULL) {
        *status = UHD_C_RX_ERROR;
    }
    if (radio == NULL || radio->rx_stream == NULL || radio->rx_md == NULL) {
        set_err(err, err_len, "RX radio is not initialized");
        return -1;
    }

    void* buffs[1] = {samples};
    size_t local_received = 0;
    uhd_error e = uhd_rx_streamer_recv(radio->rx_stream, buffs, max_samps, &radio->rx_md, timeout, false, &local_received);
    if (check_rx(e, radio->rx_stream, err, err_len) != 0) {
        return -1;
    }

    uhd_rx_metadata_error_code_t ec = UHD_RX_METADATA_ERROR_CODE_NONE;
    if (check_global(uhd_rx_metadata_error_code(radio->rx_md, &ec), err, err_len) != 0) {
        return -1;
    }

    if (received != NULL) {
        *received = local_received;
    }

    if (ec == UHD_RX_METADATA_ERROR_CODE_NONE) {
        if (status != NULL) *status = UHD_C_RX_OK;
        return 0;
    }
    if (ec == UHD_RX_METADATA_ERROR_CODE_TIMEOUT) {
        if (status != NULL) *status = UHD_C_RX_TIMEOUT;
        return 0;
    }
    if (ec == UHD_RX_METADATA_ERROR_CODE_OVERFLOW) {
        if (status != NULL) *status = UHD_C_RX_OVERFLOW;
        return 0;
    }

    char buf[2048];
    buf[0] = '\0';
    uhd_rx_metadata_strerror(radio->rx_md, buf, sizeof(buf));
    set_err(err, err_len, buf[0] ? buf : "RX metadata error");
    if (status != NULL) *status = UHD_C_RX_ERROR;
    return 0;
}

int uhd_c_pp_string(uhd_c_radio_t* radio, char* out, size_t out_len)
{
    if (radio == NULL || radio->usrp == NULL) {
        if (out != NULL && out_len > 0) {
            out[0] = '\0';
        }
        return -1;
    }
    return uhd_usrp_get_pp_string(radio->usrp, out, out_len) == UHD_ERROR_NONE ? 0 : -1;
}

int uhd_c_set_thread_priority(void)
{
    /* PXIe full-rate streaming requires the same high realtime priority as UHD benchmark_rate --priority high. */
    return uhd_set_thread_priority(1.0f, true) == UHD_ERROR_NONE ? 0 : -1;
}
