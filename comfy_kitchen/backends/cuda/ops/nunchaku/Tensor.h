#pragma once
#include "common.h"
#include <vector>
#include <cstdint>

namespace nunchaku {

struct TensorShape {
    std::vector<int> dataExtent;
    int64_t offset = 0;

    int64_t size() const {
        int64_t s = 1;
        for (int d : dataExtent) s *= d;
        return s;
    }
    int operator[](int idx) const {
        if (idx < 0) idx += dataExtent.size();
        return dataExtent[idx];
    }
};

struct Tensor {
    enum ScalarType {
        INVALID_SCALAR_TYPE,
        INT8, INT16, INT32, INT64,
        FP16, FP32, BF16,
        FP8_E4M3, FP8_E5M2
    };

    void* ptr = nullptr;
    TensorShape shape;
    ScalarType scalarType = INVALID_SCALAR_TYPE;
    
    // Minimal methods required by kernels
    __host__ __device__ bool valid() const { return ptr != nullptr; }
    __host__ __device__ int numel() const { 
        int s = 1;
        for (int d : shape.dataExtent) s *= d;
        return s;
    }
    __host__ __device__ int ndims() const { return (int)shape.dataExtent.size(); }
    __host__ __device__ ScalarType dtype() const { return scalarType; }
    __host__ __device__ ScalarType scalar_type() const { return scalarType; }

    template<typename T>
    __host__ __device__ T* data_ptr() const { return reinterpret_cast<T*>(ptr); }

    void zero_() {
        if (ptr) {
             size_t size_bytes = numel() * scalar_size(); 
             checkCUDA(cudaMemsetAsync(ptr, 0, size_bytes, getCurrentCUDAStream()));
        }
    }
    
    static size_t scalar_size(ScalarType t) {
        switch(t) {
            case INT8: return 1;
            case FP8_E4M3: return 1;
            case FP8_E5M2: return 1;
            case INT16: return 2;
            case FP16: return 2;
            case BF16: return 2;
            case INT32: return 4;
            case FP32: return 4;
            case INT64: return 8;
            default: return 1; 
        }
    }
    size_t scalar_size() const { return scalar_size(scalarType); }
    
    size_t stride(int idx) const {
         if (idx < 0) idx += ndims();
         size_t s = 1;
         for (size_t i = idx + 1; i < shape.dataExtent.size(); ++i) s *= shape.dataExtent[i];
         return s;
    }

    __host__ __device__ int size(int idx) const {
        return shape[idx];
    }
};

}
