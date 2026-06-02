#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uhd_c_radio uhd_c_radio_t;

enum {
    UHD_C_RX_OK = 0,
    UHD_C_RX_TIMEOUT = 1,
    UHD_C_RX_OVERFLOW = 2,
    UHD_C_RX_ERROR = 3
};

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
    size_t err_len);

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
    size_t err_len);

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
    size_t err_len);

void uhd_c_free(uhd_c_radio_t** radio);

size_t uhd_c_tx_max_samps(uhd_c_radio_t* radio);
size_t uhd_c_rx_max_samps(uhd_c_radio_t* radio);

int uhd_c_tx_send(
    uhd_c_radio_t* radio,
    const void* samples,
    size_t nsamps,
    int start_of_burst,
    int end_of_burst,
    double timeout,
    size_t* sent,
    char* err,
    size_t err_len);

int uhd_c_rx_start(uhd_c_radio_t* radio, char* err, size_t err_len);
int uhd_c_rx_stop(uhd_c_radio_t* radio, char* err, size_t err_len);

int uhd_c_rx_recv(
    uhd_c_radio_t* radio,
    void* samples,
    size_t max_samps,
    double timeout,
    size_t* received,
    int* status,
    char* err,
    size_t err_len);

int uhd_c_pp_string(uhd_c_radio_t* radio, char* out, size_t out_len);
int uhd_c_set_thread_priority(void);

#ifdef __cplusplus
}
#endif
