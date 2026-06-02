#include "fec_config.hpp"

#include <atomic>
#include <memory>

namespace fec {

FecConfigStore::ConfigPtr FecConfigStore::get() const
{
    return std::atomic_load_explicit(&active_, std::memory_order_acquire);
}

void FecConfigStore::swap(ConfigPtr next)
{
    std::atomic_store_explicit(&active_, std::move(next), std::memory_order_release);
}

const char* to_string(FecStandard standard)
{
    switch (standard) {
    case FecStandard::DvbS2Ldpc:
        return "dvb-s2-ldpc";
    case FecStandard::NrLdpc:
        return "5g-nr-ldpc";
    case FecStandard::Polar:
        return "polar";
    }
    return "unknown";
}

const char* to_string(NrBaseGraph bg)
{
    switch (bg) {
    case NrBaseGraph::BG1:
        return "BG1";
    case NrBaseGraph::BG2:
        return "BG2";
    }
    return "unknown";
}

}  // namespace fec
