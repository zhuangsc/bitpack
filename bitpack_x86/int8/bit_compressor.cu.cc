#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include <chrono>
#include <iostream>
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU


__global__ void BitDecompressKernelV1(const signed char* input, const int bits, const int nval, float* output) {
    typedef union {
        unsigned int i;
        float f;
    } _in_uni;
    _in_uni utmp, out;

    const int BYTE = 8;
    const int FP_LEN = BYTE * 4;
    const int MANTISSA = 23;

    const int bround = MANTISSA - bits;
    const int tot_float = nval;

    int i_idx = 0;
    for(int i = 0; i < tot_float; i++) {
        int e_bits = FP_LEN - bround;
        int acc_bits = 0;
        out.i = 0;
        while (acc_bits < FP_LEN) {
            int in_ptr = i_idx / BYTE; 
            int in_off = i_idx % BYTE;
            int len = e_bits - acc_bits;
            utmp.i = input[in_ptr];
            if (len < BYTE) {
                utmp.i = utmp.i >> (BYTE-len) << (BYTE-len);
                i_idx += len;
                break;
            }
            utmp.i = utmp.i & 0b11111111;
            int shifts = FP_LEN - acc_bits - (BYTE - in_off);
            out.i += utmp.i << shifts;
            int bits = FP_LEN-acc_bits > BYTE-in_off ? BYTE-in_off : FP_LEN-acc_bits;
            acc_bits += bits;
            i_idx += bits;
        }
        output[i] = out.f;
    }
};

__global__ void BitDecompressKernelV2(const signed char* input, const int bits, const int nval, float* output) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID >= nval)
        return ;

    typedef union {
        unsigned int i;
        float f;
    } _in_uni;
    _in_uni utmp, out;

    const int BYTE = 8;
    const int FP_LEN = BYTE * 4;
    const int MANTISSA = 23;
    const int bround = MANTISSA - bits;

    int e_bits = FP_LEN - bround;
    int i_idx = threadID * e_bits;
    int acc_bits = 0;
    out.i = 0;
    while (acc_bits < FP_LEN) {
        int in_ptr = i_idx / BYTE; 
        int in_off = i_idx % BYTE;
        int len = e_bits - acc_bits;
        utmp.i = input[in_ptr];
        if (len < BYTE) {
            utmp.i = utmp.i >> (BYTE-len) << (BYTE-len);
            //i_idx += len;
            break;
        }
        utmp.i = utmp.i & 0b11111111;
        int shifts = FP_LEN - acc_bits - (BYTE - in_off);
        out.i += utmp.i << shifts;
        int bits = FP_LEN-acc_bits > BYTE-in_off ? BYTE-in_off : FP_LEN-acc_bits;
        acc_bits += bits;
        i_idx += bits;
    }
    output[threadID] = out.f;
};

void BitDecompressLauncher(const signed char* input, const int bits, const int nval, float* output, const Eigen::GpuDevice& d) {
    auto start = std::chrono::high_resolution_clock::now();
    int threads_per_block = 1024; //NVIDIA K80
    int blocks = (nval+threads_per_block-1)/threads_per_block;

    BitDecompressKernelV2<<<blocks, threads_per_block, 0, d.stream()>>>(input, bits, nval, output);
}

typedef Eigen::GpuDevice GPUDevice;
#endif
