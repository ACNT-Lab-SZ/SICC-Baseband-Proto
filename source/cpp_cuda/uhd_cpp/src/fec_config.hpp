#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace fec {

enum class FecStandard {
    CcsdsLdpc,
    DvbS2Ldpc,
    NrLdpc,
    Polar,
};

enum class NrBaseGraph {
    BG1,
    BG2,
};

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;

    // Contiguous POD arrays intended for direct cudaMemcpy to device memory.
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_ind;
    std::vector<uint8_t> val;
};

// Type A sparse LDPC data shared by DVB-S2 and CCSDS alist profiles.
struct DvbS2LdpcParams {
    int block_length = 0;
    int information_length = 0;
    CsrMatrix h;
};

struct NrLdpcParams {
    NrBaseGraph base_graph = NrBaseGraph::BG1;
    int lifting_factor_zc = 0;
    int block_length = 0;
    int information_length = 0;
};

struct PolarCodeParams {
    int block_length = 0;
    int information_length = 0;
    std::vector<uint16_t> reliability_sequence;
};

using FecParams = std::variant<DvbS2LdpcParams, NrLdpcParams, PolarCodeParams>;

struct FecConfig {
    FecStandard standard = FecStandard::DvbS2Ldpc;
    std::string name;
    FecParams params;

    int block_length() const
    {
        return std::visit([](const auto& p) { return p.block_length; }, params);
    }

    int information_length() const
    {
        return std::visit([](const auto& p) { return p.information_length; }, params);
    }
};

class FecConfigStore {
public:
    using ConfigPtr = std::shared_ptr<const FecConfig>;

    ConfigPtr get() const;
    void swap(ConfigPtr next);

private:
    ConfigPtr active_;
};

const char* to_string(FecStandard standard);
const char* to_string(NrBaseGraph bg);

}  // namespace fec
