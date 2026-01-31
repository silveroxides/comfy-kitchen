#pragma once

#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <tuple>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <string>
#include <stdexcept>
#include <stack>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

// Minimal replacement for spdlog
namespace spdlog {
    namespace fmt_lib {
        template<typename... Args>
        std::string format(const char* fmt, Args... args) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), fmt, args...);
            return std::string(buffer);
        }
    }
    inline void trace(const char* fmt, ...) {} // No-op for trace
}

namespace nunchaku {

inline cudaError_t checkCUDA(cudaError_t retValue) {
    if (retValue != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(retValue));
        throw std::runtime_error("CUDA error");
    }
    return retValue;
}

inline thread_local std::stack<cudaStream_t> stackCUDAStreams;

inline cudaStream_t getCurrentCUDAStream() {
    if (stackCUDAStreams.empty()) {
        return 0;
    }
    return stackCUDAStreams.top();
}

struct CUDAStreamContext {
    cudaStream_t prevStream;
    CUDAStreamContext(cudaStream_t stream) {
        stackCUDAStreams.push(stream);
    }
    ~CUDAStreamContext() {
        stackCUDAStreams.pop();
    }
};

template<typename T>
constexpr T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

template<typename T>
constexpr T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Helpers for dispatch
template <typename F>
inline void dispatchBool(bool cond, F&& f) {
    if (cond) {
        f.template operator()<true>();
    } else {
        f.template operator()<false>();
    }
}

} // namespace nunchaku
