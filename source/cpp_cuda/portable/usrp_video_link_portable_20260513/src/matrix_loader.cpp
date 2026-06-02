#include "matrix_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fec {
namespace {

static CsrMatrix read_alist_as_csr(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open alist file: " + path);
    }

    int n = 0;
    int m = 0;
    in >> n >> m;
    if (!in || n <= 0 || m <= 0) {
        throw std::runtime_error("invalid alist header in: " + path);
    }

    int max_col_w = 0;
    int max_row_w = 0;
    in >> max_col_w >> max_row_w;
    if (!in || max_col_w < 0 || max_row_w < 0) {
        throw std::runtime_error("invalid alist degree header in: " + path);
    }

    std::vector<int> col_w(static_cast<size_t>(n), 0);
    std::vector<int> row_w(static_cast<size_t>(m), 0);
    for (int i = 0; i < n; ++i) {
        in >> col_w[static_cast<size_t>(i)];
    }
    for (int r = 0; r < m; ++r) {
        in >> row_w[static_cast<size_t>(r)];
    }

    // Column adjacency appears first in alist. It is not needed for CSR, but
    // must be consumed before row adjacency.
    for (int c = 0; c < n; ++c) {
        for (int j = 0; j < max_col_w; ++j) {
            int tmp = 0;
            in >> tmp;
        }
    }

    std::vector<std::vector<int32_t>> rows(static_cast<size_t>(m));
    for (int r = 0; r < m; ++r) {
        auto& row = rows[static_cast<size_t>(r)];
        row.reserve(static_cast<size_t>(std::max(row_w[static_cast<size_t>(r)], 0)));
        for (int j = 0; j < max_row_w; ++j) {
            int one_based_col = 0;
            in >> one_based_col;
            if (one_based_col > 0) {
                row.push_back(static_cast<int32_t>(one_based_col - 1));
            }
        }
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    CsrMatrix csr;
    csr.rows = m;
    csr.cols = n;
    csr.row_ptr.resize(static_cast<size_t>(m + 1), 0);
    for (int r = 0; r < m; ++r) {
        csr.row_ptr[static_cast<size_t>(r + 1)] =
            csr.row_ptr[static_cast<size_t>(r)] +
            static_cast<int32_t>(rows[static_cast<size_t>(r)].size());
    }
    csr.nnz = csr.row_ptr.back();
    csr.col_ind.reserve(static_cast<size_t>(csr.nnz));
    csr.val.reserve(static_cast<size_t>(csr.nnz));
    for (const auto& row : rows) {
        for (int32_t col : row) {
            csr.col_ind.push_back(col);
            csr.val.push_back(1);
        }
    }

    return csr;
}

static std::vector<uint16_t> read_reliability_sequence(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open polar reliability sequence: " + path);
    }

    std::vector<uint16_t> seq;
    std::string token;
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string discard;
            std::getline(in, discard);
            continue;
        }
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) {
                        return c == ',' || c == ';';
                    }),
                    token.end());
        if (token.empty()) {
            continue;
        }
        const int idx = std::stoi(token);
        if (idx < 0 || idx > 65535) {
            throw std::runtime_error("polar reliability index out of uint16 range");
        }
        seq.push_back(static_cast<uint16_t>(idx));
    }
    return seq;
}

}  // namespace

MatrixLoadRequest MatrixLoadRequest::dvb_s2_alist(
    std::string alist_path,
    int information_length,
    std::string name)
{
    MatrixLoadRequest r;
    r.standard = FecStandard::DvbS2Ldpc;
    r.path = std::move(alist_path);
    r.information_length = information_length;
    r.name = std::move(name);
    return r;
}

MatrixLoadRequest MatrixLoadRequest::nr_ldpc(
    NrBaseGraph bg,
    int zc,
    int block_length,
    int information_length,
    std::string name)
{
    MatrixLoadRequest r;
    r.standard = FecStandard::NrLdpc;
    r.nr_base_graph = bg;
    r.nr_lifting_factor_zc = zc;
    r.nr_block_length = block_length;
    r.nr_information_length = information_length;
    r.name = std::move(name);
    return r;
}

MatrixLoadRequest MatrixLoadRequest::polar(
    std::string reliability_sequence_path,
    int block_length,
    int information_length,
    std::string name)
{
    MatrixLoadRequest r;
    r.standard = FecStandard::Polar;
    r.path = std::move(reliability_sequence_path);
    r.polar_block_length = block_length;
    r.polar_information_length = information_length;
    r.name = std::move(name);
    return r;
}

MatrixLoader::ConfigPtr MatrixLoader::load(const MatrixLoadRequest& request) const
{
    switch (request.standard) {
    case FecStandard::DvbS2Ldpc:
        return load_dvb_s2_alist(request.path, request.information_length, request.name);
    case FecStandard::NrLdpc:
        return load_nr_ldpc(
            request.nr_base_graph,
            request.nr_lifting_factor_zc,
            request.nr_block_length,
            request.nr_information_length,
            request.name);
    case FecStandard::Polar:
        return load_polar(
            request.path,
            request.polar_block_length,
            request.polar_information_length,
            request.name);
    }
    throw std::runtime_error("unsupported FEC standard");
}

MatrixLoader::ConfigPtr MatrixLoader::load_dvb_s2_alist(
    const std::string& path,
    int information_length,
    const std::string& name) const
{
    DvbS2LdpcParams p;
    p.h = read_alist_as_csr(path);
    p.block_length = p.h.cols;
    p.information_length = information_length > 0 ? information_length : (p.h.cols - p.h.rows);
    if (p.information_length <= 0 || p.information_length >= p.block_length) {
        throw std::runtime_error("invalid DVB-S2 LDPC dimensions after loading: " + path);
    }

    auto cfg = std::make_shared<FecConfig>();
    cfg->standard = FecStandard::DvbS2Ldpc;
    cfg->name = name.empty() ? path : name;
    cfg->params = std::move(p);
    return cfg;
}

MatrixLoader::ConfigPtr MatrixLoader::load_nr_ldpc(
    NrBaseGraph bg,
    int lifting_factor_zc,
    int block_length,
    int information_length,
    const std::string& name) const
{
    if (lifting_factor_zc <= 0 || block_length <= 0 || information_length <= 0 ||
        information_length >= block_length) {
        throw std::runtime_error("invalid 5G NR LDPC dimensions");
    }

    NrLdpcParams p;
    p.base_graph = bg;
    p.lifting_factor_zc = lifting_factor_zc;
    p.block_length = block_length;
    p.information_length = information_length;

    auto cfg = std::make_shared<FecConfig>();
    cfg->standard = FecStandard::NrLdpc;
    cfg->name = name.empty() ? std::string("5g-nr-ldpc") : name;
    cfg->params = p;
    return cfg;
}

MatrixLoader::ConfigPtr MatrixLoader::load_polar(
    const std::string& reliability_sequence_path,
    int block_length,
    int information_length,
    const std::string& name) const
{
    if (block_length <= 0 || information_length <= 0 || information_length >= block_length) {
        throw std::runtime_error("invalid polar dimensions");
    }
    PolarCodeParams p;
    p.block_length = block_length;
    p.information_length = information_length;
    p.reliability_sequence = read_reliability_sequence(reliability_sequence_path);
    if (static_cast<int>(p.reliability_sequence.size()) < block_length) {
        throw std::runtime_error("polar reliability sequence is shorter than N");
    }

    auto cfg = std::make_shared<FecConfig>();
    cfg->standard = FecStandard::Polar;
    cfg->name = name.empty() ? reliability_sequence_path : name;
    cfg->params = std::move(p);
    return cfg;
}

}  // namespace fec
