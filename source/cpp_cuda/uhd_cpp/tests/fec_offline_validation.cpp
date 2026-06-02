#include "matrix_loader.hpp"
#include "rx_decoder_router.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct OfdmShape {
    int nfft = 1024;
    int active_sc = 512;
    int num_symbols = 48;
    int pilot_period = 4;
    int bits_per_symbol = 2;
};

static int data_symbols(const OfdmShape& s)
{
    if (s.pilot_period <= 0) {
        return s.num_symbols - 3;
    }
    int pilots = 0;
    for (int sym = 0; sym < s.num_symbols; sym += s.pilot_period) {
        ++pilots;
    }
    return s.num_symbols - pilots;
}

static int coded_bits_per_frame(const OfdmShape& s)
{
    return s.active_sc * data_symbols(s) * s.bits_per_symbol;
}

static void require(bool ok, const std::string& msg)
{
    if (!ok) {
        throw std::runtime_error(msg);
    }
}

static std::string join_path(const char* root, const std::string& rel)
{
    std::string out = root ? root : ".";
    if (!out.empty() && out.back() != '\\' && out.back() != '/') {
        out += "\\";
    }
    out += rel;
    return out;
}

static void print_pass(const std::string& name)
{
    std::cout << "[PASS] " << name << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const char* repo = argc > 1 ? argv[1] : std::getenv("USRP_PLT_ROOT");
        if (repo == nullptr || std::string(repo).empty()) {
            repo = ".";
        }

        fec::MatrixLoader loader;
        fec::RxDecoderRuntimeOptions route_opts;
        route_opts.requested_decoder = "auto";

        const auto ccsds512 = loader.load_ccsds_alist(
            join_path(repo, "data\\code_matrices\\Code_Matrices_Lib\\LDPC\\CCSDS_ldpc_n512_k256.alist"),
            256,
            "ccsds-n512-k256");
        require(ccsds512->standard == fec::FecStandard::CcsdsLdpc, "CCSDS standard label mismatch");
        require(ccsds512->block_length() == 512, "CCSDS n512 block length mismatch");
        require(ccsds512->information_length() == 256, "CCSDS n512 information length mismatch");
        const auto ccsds512_plan = fec::RxDecoderRouter::make_plan(ccsds512, route_opts);
        require(ccsds512_plan.kind == fec::RxDecoderKind::CpuMinSum, "CCSDS n512 should auto-route to CPU min-sum");
        print_pass("Type A CSR load and short-code route: CCSDS n512/k256 -> CPU min-sum");

        const auto dvbs2 = loader.load_dvb_s2_alist(
            join_path(repo, "data\\code_matrices\\Code_Matrices_Lib\\LDPC\\DVB_S2_N64800_R12.alist"),
            32400,
            "dvb-s2-n64800-r12");
        require(dvbs2->block_length() == 64800, "DVB-S2 n64800 block length mismatch");
        require(dvbs2->information_length() == 32400, "DVB-S2 n64800 information length mismatch");
        const auto dvbs2_plan = fec::RxDecoderRouter::make_plan(dvbs2, route_opts);
        require(dvbs2_plan.kind == fec::RxDecoderKind::GpuBp, "DVB-S2 N > 1000 must force GPU BP");
        require(dvbs2_plan.requires_gpu, "DVB-S2 long-code route must require GPU");
        print_pass("Type A CSR load and long-code route: DVB-S2 n64800/k32400 -> forced GPU BP");

        const auto nr = loader.load_nr_ldpc(fec::NrBaseGraph::BG1, 384, 25344, 8448, "nr-bg1-zc384");
        require(nr->block_length() == 25344, "NR block length mismatch");
        const auto nr_plan = fec::RxDecoderRouter::make_plan(nr, route_opts);
        require(nr_plan.kind == fec::RxDecoderKind::GpuBp, "NR long LDPC must force GPU BP");
        print_pass("Type B QC-LDPC params and route: NR BG1 Zc384 -> forced GPU BP");

        const auto polar = loader.load_polar(
            join_path(repo, "uhd_cpp\\tests\\polar_q_16.txt"),
            16,
            8,
            "polar-n16-k8");
        require(polar->block_length() == 16, "Polar block length mismatch");
        const auto polar_plan = fec::RxDecoderRouter::make_plan(polar, route_opts);
        require(polar_plan.kind == fec::RxDecoderKind::PolarScl, "Short Polar should route to SCL");
        print_pass("Type C Polar reliability sequence load and route: N16/K8 -> SCL");

        const OfdmShape current{};
        const int current_capacity = coded_bits_per_frame(current);
        require(current_capacity == 36864, "current OFDM capacity changed unexpectedly");
        require(current_capacity < 64800, "default OFDM capacity should not fit DVB-S2 N64800");
        print_pass("OFDM capacity check: default frame 36864 coded bits, does not fit DVB-S2 N64800");

        OfdmShape dvbs2_normal = current;
        dvbs2_normal.active_sc = 900;
        const int dvbs2_capacity = coded_bits_per_frame(dvbs2_normal);
        require(dvbs2_capacity == 64800, "active_sc=900 OFDM capacity should fit DVB-S2 N64800 exactly");
        print_pass("OFDM capacity check: active_sc=900 gives 64800 coded bits for one DVB-S2 normal frame");

        std::cout << "[PASS] Offline FEC architecture validation completed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
