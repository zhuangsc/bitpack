#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include <cstdio>
#include <cstdlib>
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU


__global__ void BitUnpackKernel(const float* input, const int bits, const int nval, float* output) {
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
    int bround = MANTISSA - bits;
    bround = bits;

    int e_bits = FP_LEN - bround;
    int i_idx = threadID * e_bits;
    out.i = 0;
    int acc_bits = 0;
    while (acc_bits < e_bits) {
        int in_ptr = i_idx / FP_LEN; 
        int in_off = i_idx % FP_LEN;
        //int res_len = e_bits - acc_bits;
        int in_len = FP_LEN - in_off;
        utmp.f = input[in_ptr];
        out.i += utmp.i << in_off >> acc_bits;
        i_idx += in_len;
        acc_bits += in_len;
    }
    output[threadID] = out.f;
};

void BitUnpackLauncher(const float* input, const int bits, const int nval, float* output, const Eigen::GpuDevice& d) {
    int threads_per_block = 512; //NVIDIA Kepler
    int blocks = (nval+threads_per_block-1)/threads_per_block;

    BitUnpackKernel<<<blocks, threads_per_block, 0, d.stream()>>>(input, bits, nval, output);
}


__global__ void BitUnpackGpuKernel(const float* input, const int bits, const int nval, float* output) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID >= nval)
        return;

    typedef union {
        unsigned char *c;
        unsigned int *i;
        float *f;
    } _in_unip;
    _in_unip srcp, dstp;

    const int BYTE = 8;
    const int FP_BYTES = 4;
    const int MANTISSA = 23;
    const int FP_LEN = BYTE * FP_BYTES;
    const int bround = MANTISSA - bits;
    int ebits = FP_LEN - bround;
    ebits = bits;

    const int int8_per_float = (ebits + BYTE -1) / BYTE;
    const int tot_float_out = nval;
    const int tot_int8_out = tot_float_out * FP_BYTES;
    const int tot_int8_in = tot_float_out * int8_per_float;
    const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

    srcp.c = (unsigned char *) input + threadID * int8_per_float;
    dstp.f = output + threadID;
    for( int i = 0; i < FP_BYTES; i++ ) {
        unsigned char *dst = dstp.c + i;
        if ( i >= (FP_BYTES-int8_per_float) ) {
            unsigned char *src = srcp.c;
            *dst = *src;
            srcp.c += 1;
        } else {
           *dst = 0;
        }
    }

    unsigned int aux = ~0;
    aux <<= FP_LEN - ebits;
    *dstp.i = *dstp.i & aux;
};

void BitUnpackGpuLauncher(const float* input, const int bits, const int nval, float* output, const Eigen::GpuDevice& d) {
    int threads_per_block = 512; //NVIDIA Kepler
    int blocks = (nval+threads_per_block-1)/threads_per_block;

    BitUnpackGpuKernel<<<blocks, threads_per_block, 0, d.stream()>>>(input, bits, nval, output);
    //typedef union {
    //    unsigned int *i;
    //    float *f;
    //} _in_unip;
    //_in_unip val;
    //val.i = (unsigned int*) std::malloc(sizeof(float)*nval);
    //cudaMemcpy(val.i, output, sizeof(float)*nval, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < nval; i++)
    //    std::printf("0x%X  ", val.i[i]);
    //std::printf("\n");
    //std::free(val.i);
}


__global__ void BitPackGpuKernel(const float* input_eigen, const int bits, const int nval_in, float* output_eigen) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID >= nval_in)
        return;

    const int BYTE = 8;
    const int FP_BYTES = 4;
    const int FP_LEN = BYTE * FP_BYTES;
    const int MANTISSA = 23;
    const int bround = MANTISSA - bits;
    int ebits = FP_LEN - bround;
    ebits = bits;

    const int tot_float_in = nval_in;
    const int tot_int8_in = tot_float_in * FP_BYTES;
    const int int8_per_float = (ebits + BYTE - 1) / BYTE;
    const int tot_int8_out = tot_float_in * int8_per_float;
    const int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

    typedef union {
        unsigned char *c; 
        unsigned int *i;
        float *f;
    } _in_unip;
    _in_unip input, output, src, dst;

    input.f = (float *) input_eigen;
    output.f = output_eigen;

    int skip = FP_BYTES - int8_per_float;
    src.f = input.f + threadID;
    src.c = src.c + skip;
    dst.c = output.c + (threadID*int8_per_float);
    for(int i = 0; i < int8_per_float; i++) {
        *dst.c = *src.c;
        src.c += 1;
        dst.c += 1;
    }

    //for(int i = 0; i < tot_float_in; i++) {
    //    int skip = FP_BYTES - int8_per_float;
    //    src.f = input.f + i;
    //    src.c = src.c + skip;
    //    dst.c = output.c + (i*int8_per_float);
    //    int chunk_size = int8_per_float;
    //    std::memcpy(dst.c, src.c, chunk_size);
    //}
};

void BitPackGpuLauncher(const float* input, const int bits, const int nval_in, float* output, const Eigen::GpuDevice& d) {
    int threads_per_block = 512; //NVIDIA Kepler
    int blocks = (nval_in+threads_per_block-1)/threads_per_block;

    BitPackGpuKernel<<<blocks, threads_per_block, 0, d.stream()>>>(input, bits, nval_in, output);
    //typedef union {
    //    unsigned int *i;
    //    float *f;
    //} _in_unip;
    //_in_unip val;
    //val.i = (unsigned int*) std::malloc(sizeof(float)*nval);
    //cudaMemcpy(val.i, output, sizeof(float)*nval, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < nval; i++)
    //    std::printf("0x%X  ", val.i[i]);
    //std::printf("\n");
    //std::free(val.i);
}

typedef Eigen::GpuDevice GPUDevice;

#endif
