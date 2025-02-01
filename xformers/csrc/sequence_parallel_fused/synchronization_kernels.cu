#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda/std/chrono>
#include <cuda/atomic>

#define CUDA_ARCH_SUPPORTS_ATOMICS (__CUDA_ARCH__ >= 700 || (!defined(_MSC_VER) && __CUDA_ARCH__ >= 600))
#if CUDA_ARCH_SUPPORTS_ATOMICS
#include <cuda/std/version>
#define LIBCUDACXX_PROVIDES_ATOMIC_REF (_LIBCUDACXX_CUDA_API_VERSION >= 001007000)
#endif

constexpr int kMaxWorldSize = 8;
constexpr int kNumSpinsBetweenTimeoutChecks = 1000;

__device__ uint64_t getNsSinceEpoch() {
    return cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
               cuda::std::chrono::system_clock::now().time_since_epoch())
        .count();
}

#if CUDA_ARCH_SUPPORTS_ATOMICS
#if LIBCUDACXX_PROVIDES_ATOMIC_REF
class Atomic {
 public:
  __device__ explicit Atomic(int* ptr) : ref_(*reinterpret_cast<cuda::atomic_ref<int, cuda::thread_scope_system>*>(ptr)) {}
  __device__ int load() { return ref_.load(cuda::std::memory_order_acquire); }
  __device__ void store(int val) { ref_.store(val, cuda::std::memory_order_release); }
 private:
  cuda::atomic_ref<int, cuda::thread_scope_system> ref_;
};
#else
class Atomic {
 public:
  __device__ explicit Atomic(int* ptr) : ref_(*reinterpret_cast<cuda::atomic<int, cuda::thread_scope_system>*>(ptr)) {}
  __device__ int load() { return ref_.load(cuda::std::memory_order_acquire); }
  __device__ void store(int val) { ref_.store(val, cuda::std::memory_order_release); }
 private:
  cuda::atomic<int, cuda::thread_scope_system>& ref_;
};
#endif
#else
class Atomic {
 public:
  __device__ explicit Atomic(int* ptr) : ptr_(ptr) {}
  __device__ int load() {
    int val;
    asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(val) : "l"(ptr_));
    return val;
  }
  __device__ void store(int val) {
    asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(ptr_), "r"(val));
  }
 private:
  int* ptr_;
};
#endif

__global__ void write_values_kernel(const int* ptrs[], size_t numPtrs, int seqNum) {
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < numPtrs; i++) {
            Atomic atomic(ptrs[i]);
            atomic.store(seqNum);
        }
    }
}

__global__ void wait_values_kernel(const int* ptrs[], size_t numPtrs, int seqNum, uint64_t timeoutNs) {
    uint64_t startTimeNs = getNsSinceEpoch();
    for (size_t i = 0; i < numPtrs; i++) {
        Atomic atomic(ptrs[i]);
        uint64_t numSpins = 0;
        while (atomic.load() != seqNum) {
            numSpins++;
            if (numSpins == kNumSpinsBetweenTimeoutChecks && getNsSinceEpoch() - startTimeNs >= timeoutNs) {
                asm volatile("trap;");
            }
        }
    }
}

void write_values(torch::TensorList targets, torch::Scalar value, c10::Stream stream) {
    std::array<int*, kMaxWorldSize> rawTargets;
    for (size_t i = 0; i < targets.size(); i++) {
        rawTargets[i] = targets[i].data_ptr<int>();
    }
    write_values_kernel<<<1, 1, 0, c10::cuda::CUDAStream(stream)>>>(rawTargets.data(), targets.size(), static_cast<int>(value.toLong()));
    C10_CUDA_CHECK(cudaGetLastError());
}

void wait_values(torch::TensorList sources, torch::Scalar value, c10::Stream stream, torch::Scalar timeoutS) {
    std::array<int*, kMaxWorldSize> rawSources;
    for (size_t i = 0; i < sources.size(); i++) {
        rawSources[i] = sources[i].data_ptr<int>();
    }
    wait_values_kernel<<<1, 1, 0, c10::cuda::CUDAStream(stream)>>>(rawSources.data(), sources.size(), static_cast<int>(value.toLong()), static_cast<uint64_t>(timeoutS.toLong()) * 1000000000);
    C10_CUDA_CHECK(cudaGetLastError());
}

