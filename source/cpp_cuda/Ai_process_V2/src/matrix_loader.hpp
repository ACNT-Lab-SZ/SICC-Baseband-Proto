#pragma once

#include "fec_config.hpp"

#include <memory>
#include <string>

namespace fec {

struct MatrixLoadRequest {
    FecStandard standard = FecStandard::DvbS2Ldpc;
    std::string path;
    std::string name;

    // DVB-S2 / sparse-LDPC metadata. If information_length is 0, the loader
    // derives K as N-M for a full-rank rate-compatible parity-check matrix.
    int information_length = 0;

    // 5G NR LDPC metadata. The loader keeps only BG/Zc; it does not expand H.
    NrBaseGraph nr_base_graph = NrBaseGraph::BG1;
    int nr_lifting_factor_zc = 0;
    int nr_block_length = 0;
    int nr_information_length = 0;

    // Polar metadata. path points to a reliability-sequence text file.
    int polar_block_length = 0;
    int polar_information_length = 0;

    static MatrixLoadRequest dvb_s2_alist(
        std::string alist_path,
        int information_length = 0,
        std::string name = {});

    static MatrixLoadRequest nr_ldpc(
        NrBaseGraph bg,
        int zc,
        int block_length,
        int information_length,
        std::string name = {});

    static MatrixLoadRequest polar(
        std::string reliability_sequence_path,
        int block_length,
        int information_length,
        std::string name = {});
};

class MatrixLoader {
public:
    using ConfigPtr = std::shared_ptr<const FecConfig>;

    ConfigPtr load(const MatrixLoadRequest& request) const;

    ConfigPtr load_dvb_s2_alist(
        const std::string& path,
        int information_length = 0,
        const std::string& name = {}) const;

    ConfigPtr load_nr_ldpc(
        NrBaseGraph bg,
        int lifting_factor_zc,
        int block_length,
        int information_length,
        const std::string& name = {}) const;

    ConfigPtr load_polar(
        const std::string& reliability_sequence_path,
        int block_length,
        int information_length,
        const std::string& name = {}) const;
};

}  // namespace fec
