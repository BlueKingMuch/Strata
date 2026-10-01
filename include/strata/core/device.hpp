// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE cudaMalloc per planner region with bump sub-allocation below it and no frees.  That is
// not a simplification for the first version: the memory plan from P1.S9 is fixed at startup, so the set of
// regions and their sizes is known before anything is allocated, and an allocator that can free would be
// solving a problem the engine does not have while adding fragmentation and failure modes it does.
//
// The reason to get this in early is that the VRAM budget is the binding constraint of the whole design
// (5.95 GB pooled between KV and the expert cache, 33.97 GB of experts in DRAM).  A runtime that discovers at
// token 4000 that it has overcommitted has already lost; the plan is printed against `cudaMemGetInfo` at
// startup so the discrepancy is visible immediately.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
    std::string arch;              // HIP: gcnArchName without its feature suffix ("gfx1201"); empty on CUDA
};

// HIP builds: whether GPU `ordinal` can run this binary - its architecture must be one the binary was COMPILED
// for (STRATA_HIP_ARCHS, set by cmake/hip_backend.cmake) and it must run wave32.  "" when it can (or when there
// is no such device: the caller's own device errors apply), else the reason in a sentence.  A binary carried to
// another card would otherwise fail later with "invalid device function".  CUDA builds: always "".
std::string gpu_arch_problem(int ordinal);

// The GPU architectures this binary was compiled for ("gfx1100,gfx1201"); "" on CUDA builds.
const char* compiled_gpu_archs();

// Windows HIP builds: the private memory ("scratch") per thread that the kernels of a captured graph need - the most
// any of its kernel nodes needs (`unknown`: kernel nodes whose need could not be read) - and a launch on `stream`
// that has the runtime allocate at least that much now.  The Windows runtime sizes its scratch buffer while it
// ENQUEUES a kernel, and before it grows the buffer it waits until the GPU has finished the work in flight that
// used the old one.  A graph whose kernels wait on the GPU for the CPU (a verify window, the token graph) then
// hangs in its own launch: the GPU waits for a flag the CPU raises only once the launch has returned (RX 6800, HIP
// SDK 7.2: the first verify window of a process).  The buffer only grows, so reserving once per stream before the
// first launch, while nothing waits, is enough.  CUDA and Linux HIP builds: 0, and no launch.
size_t graph_scratch_bytes(void* graph, int* unknown = nullptr);
bool reserve_scratch(size_t bytes_per_thread, void* stream, std::string& err, size_t* reserved = nullptr);

// Throws when there is no CUDA device.  The engine targets sm_120 specifically and must say so rather than
// run slowly on something else: `CMakeLists.txt` already refuses to COMPILE for another architecture, and
// this is the matching check at run time (a binary can be carried to a different machine).
DeviceInfo device_info(int ordinal = 0);

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// One cudaMalloc, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a debug
// build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same reasoning as
// the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

}  // namespace strata::core
