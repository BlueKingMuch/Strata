// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/core/device.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace strata::core {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw CudaError(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
// "gfx1201:sramecc-:xnack-" -> "gfx1201"
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

#if defined(STRATA_USE_HIP) && defined(_WIN32)
namespace {
// N ints of private memory per thread: volatile and indexed at run time, so they stay in scratch at every optimization
// level.  `out` is never written (the launch passes i = 1).  Its few registers let the runtime size the buffer for
// the most waves per CU, so it covers any kernel that needs no more private memory per thread.
template <int N> __global__ void scratch_reserve_kernel(int i, int* out) {
    volatile int buf[N];
    buf[i & (N - 1)] = i;
    if (i < 0) *out = buf[(i + 1) & (N - 1)];
}
constexpr int kReserveSizes = 6;   // 8 .. 256 ints: 32 B .. 1 KiB per thread
const void* reserve_fn(int k) {
    switch (k) {
    case 0: return (const void*) scratch_reserve_kernel<8>;
    case 1: return (const void*) scratch_reserve_kernel<16>;
    case 2: return (const void*) scratch_reserve_kernel<32>;
    case 3: return (const void*) scratch_reserve_kernel<64>;
    case 4: return (const void*) scratch_reserve_kernel<128>;
    default: return (const void*) scratch_reserve_kernel<256>;
    }
}
void reserve_launch(int k, hipStream_t s) {
    switch (k) {
    case 0: scratch_reserve_kernel<8><<<1, 32, 0, s>>>(1, nullptr); break;
    case 1: scratch_reserve_kernel<16><<<1, 32, 0, s>>>(1, nullptr); break;
    case 2: scratch_reserve_kernel<32><<<1, 32, 0, s>>>(1, nullptr); break;
    case 3: scratch_reserve_kernel<64><<<1, 32, 0, s>>>(1, nullptr); break;
    case 4: scratch_reserve_kernel<128><<<1, 32, 0, s>>>(1, nullptr); break;
    default: scratch_reserve_kernel<256><<<1, 32, 0, s>>>(1, nullptr); break;
    }
}
}  // namespace

size_t graph_scratch_bytes(void* graph, int* unknown) {
    if (unknown != nullptr) *unknown = 0;
    size_t n = 0;
    if (graph == nullptr || hipGraphGetNodes((hipGraph_t) graph, nullptr, &n) != hipSuccess || n == 0) {
        (void) hipGetLastError();
        return 0;
    }
    std::vector<hipGraphNode_t> nodes(n);
    if (hipGraphGetNodes((hipGraph_t) graph, nodes.data(), &n) != hipSuccess) {
        (void) hipGetLastError();
        return 0;
    }
    std::unordered_set<const void*> seen;   // a window launches the same few dozen kernels in every layer
    size_t most = 0;
    for (size_t i = 0; i < n; ++i) {
        hipGraphNodeType type;
        if (hipGraphNodeGetType(nodes[i], &type) != hipSuccess || type != hipGraphNodeTypeKernel) continue;
        hipKernelNodeParams p{};
        if (hipGraphKernelNodeGetParams(nodes[i], &p) != hipSuccess || p.func == nullptr) {
            if (unknown != nullptr) ++*unknown;
            continue;
        }
        if (!seen.insert(p.func).second) continue;
        hipFuncAttributes a{};
        if (hipFuncGetAttributes(&a, p.func) != hipSuccess) {
            if (unknown != nullptr) ++*unknown;
            continue;
        }
        most = std::max(most, (size_t) a.localSizeBytes);
    }
    (void) hipGetLastError();   // a failed lookup leaves no error behind for the next launch to report
    return most;
}

bool reserve_scratch(size_t bytes_per_thread, void* stream, std::string& err, size_t* reserved) {
    if (reserved != nullptr) *reserved = 0;
    // STRATA_SCRATCH_RESERVE=0: no reserve (the A/B; a window whose kernels need more scratch than anything run
    // before then hangs in its first launch)
    static const bool off = [] {
        const char* v = std::getenv("STRATA_SCRATCH_RESERVE");
        return v != nullptr && std::atoi(v) == 0;
    }();
    if (bytes_per_thread == 0 || off) return true;
    int k = 0;
    size_t have = 0;
    for (; k < kReserveSizes; ++k) {   // the smallest reserve kernel that needs at least as much
        hipFuncAttributes a{};
        if (hipFuncGetAttributes(&a, reserve_fn(k)) != hipSuccess) {
            (void) hipGetLastError();
            continue;
        }
        have = (size_t) a.localSizeBytes;
        if (have >= bytes_per_thread) break;
    }
    if (k == kReserveSizes) k = kReserveSizes - 1;   // more than 1 KiB per thread: the largest, and the caller says so
    reserve_launch(k, (hipStream_t) stream);
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipStreamSynchronize((hipStream_t) stream);
    if (e != hipSuccess) {
        err = std::string("reserving ") + std::to_string(bytes_per_thread) + " B of scratch per thread: " +
              hipGetErrorString(e);
        return false;
    }
    if (reserved != nullptr) *reserved = have;
    return true;
}
#else
size_t graph_scratch_bytes(void*, int* unknown) {
    if (unknown != nullptr) *unknown = 0;
    return 0;
}
bool reserve_scratch(size_t, void*, std::string&, size_t* reserved) {
    if (reserved != nullptr) *reserved = 0;
    return true;
}
#endif

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
#if defined(STRATA_USE_HIP)
        throw CudaError(std::string("no HIP device is present; this engine was compiled for ") + STRATA_HIP_ARCHS, -1);
#else
        throw CudaError("no CUDA device is present; Strata needs an NVIDIA GPU (RTX 20 series or newer)", -1);
#endif
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    check(cudaSetDevice(ordinal), "cudaSetDevice");

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;

    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");

    // The engine supports compute capability 7.5 and newer (Turing: the QSA scorer's tf32 mma has a portable
    // fp32-FMA fallback below sm_80, the tensor-core prompt kernels refuse and fall back).  Compiling for a
    // supported arch is enforced by CMake; RUNNING on an older card is caught here, because a binary can be carried
    // to a machine with an older card and would otherwise silently take whatever path the driver chose.  The HIP
    // backend checks the card against the architectures the binary was compiled for (and wave32).
#if defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#else
    // #236: the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON: Pascal sm_60, Volta sm_70) runs on the cards it
    // was built for - refusing them below 7.5 there made the flag useless; the release engine keeps 7.5
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr int kMinCc = 60;
    const char* const kNeed = "6.0 or newer (this is the experimental Pascal / Volta build)";
#else
    constexpr int kMinCc = 75;
    const char* const kNeed = "7.5 or newer (RTX 20 / 30 / 40 / 50 series)";
#endif
    if (d.cc_major * 10 + d.cc_minor < kMinCc) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata needs compute capability " + kNeed,
                        -1);
    }
#endif
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        // gridDim.x is 32-bit, so a large region needs a loop.  12 GB of floats is 3e9 elements = 1.2e7
        // blocks, which fits, but the loop keeps it correct for any size rather than for today's sizes.
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads);
            check(cudaGetLastError(), "poison_kernel");
        }
        check(cudaDeviceSynchronize(), "poison sync");
    }
}

DeviceArena::~DeviceArena() {
    if (base_) cudaFree(base_);          // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
