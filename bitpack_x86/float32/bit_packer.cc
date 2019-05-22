#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <pthread.h>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <immintrin.h>
#include <pmmintrin.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

void BitUnpackLauncher(const float* input, const int bits, const int nval, float* output, const GPUDevice& d);
void BitUnpackGpuLauncher(const float* input, const int bits, const int nval, float* output, const GPUDevice& d);
void BitPackGpuLauncher(const float* input, const int bits, const int nval_in, float* output, const GPUDevice& d);

/*
REGISTER_OP("BitPack")
    .Input("input: float32")
    .Input("bits: int32")
    .Output("packed: float32");

class BitPackOp : public OpKernel {
    public:
        explicit BitPackOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned int i; //Logical shift requires unsigned type
                float f;
            } _in_uni;
            _in_uni utmpi, utmpo;

            const Tensor& traw = context->input(0);
            auto input = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const int BYTE = 8;
            const int FP_LEN = BYTE * 4;
            const int MANTISSA = 23;

            const int N = input.size();
            int bround = MANTISSA - bits(0);
            bround = bits(0);
			int tot_bits_in = N * FP_LEN;
            int tot_bits_out = N * (FP_LEN-bround);
            int tot_float32 = (tot_bits_out+FP_LEN-1)/FP_LEN;
            if (sizeof(int) != 4)
                std::printf("WARNING!!! sizeof int: %lu\n", sizeof(int));
#ifdef __DEBUG__
                std::printf("N: %d, tot_in_bits: %d, tot_out_bits: %d, tot_float32: %d\n", N, tot_bits_in, tot_bits_out, tot_float32);
#endif
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float32}), &output_tensor));

            auto output = output_tensor->flat<float>();

            int num_threads = 0;
            int i, j;
            int workers = omp_get_max_threads();
            #pragma omp parallel for shared(input, output, num_threads) private(i, j, utmpi, utmpo) //schedule(static)
            for(j = 0; j < workers; j++) {
#ifdef __OPENMP__
                int tid = omp_get_thread_num();
                if (tid == 0)
                    num_threads = omp_get_num_threads();
#endif
                int chunk_size = 0;
                if ( j < workers-1)
                    chunk_size = tot_float32 / workers;
                else
                    chunk_size = tot_float32-(tot_float32/workers)*(workers-1);

                for(i = 0; i < chunk_size; i++) {
                    int index = j*(tot_float32/workers)+i;
                    int e_bits = FP_LEN - bround;
                    int stored_bits = index * FP_LEN;
                    int in_ptr = stored_bits / e_bits;
                    int in_off = stored_bits % e_bits;
                    int i_idx = in_ptr * FP_LEN + in_off;

                    int idx = 0;
                    int acc_bits = 0;
                    while (acc_bits < FP_LEN) {
                        in_ptr = i_idx / FP_LEN;
                        in_off = i_idx % FP_LEN;
                        int in_len = FP_LEN - in_off - bround;
                        int remain = FP_LEN - acc_bits;
                        int bits = in_len <= remain ? in_len : remain;
                        utmpi.f = input(in_ptr);
                        if (acc_bits == 0) {
                            utmpo.i = 0;
                        } else {
                            utmpo.f = output(index);
                        }
                        utmpo.i += (utmpi.i >> bround << (bround+in_off) >> acc_bits);
                        output(index) = utmpo.f;
                        acc_bits += bits;
                        i_idx = bits >= in_len ? (in_ptr+1)*FP_LEN : i_idx+bits;
                    }
                }
            }
            //std::printf("Avaiable processors: %d\n", num_threads);

#ifdef __DEBUG__
            std::printf("size: %d\n", output.size());
            for(int i = 0; i < tot_float32; i++){
                utmpo.f = output(i);
                std::printf("0x%X\n", utmpo.i);
            }
#endif
        }
};

class BitCompressOp : public OpKernel {
    public:
        explicit BitCompressOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned int i; //Logical shift requires unsigned type
                float f;
            } _in_uni;
            _in_uni utmpi, utmpo;

            const Tensor& traw = context->input(0);
            auto input = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const int BYTE = 8;
            const int FP_LEN = BYTE * 4;
            const int MANTISSA = 23;

            const int N = input.size();
            const int bround = MANTISSA - bits(0);
			int tot_bits_in = N * FP_LEN;
            int tot_bits_out = N * (FP_LEN-bround);
            int tot_float32 = (tot_bits_out+FP_LEN-1)/FP_LEN;
            if (sizeof(int) != 4)
                std::printf("WARNING!!! sizeof int: %lu\n", sizeof(int));
#ifdef __DEBUG__
                std::printf("N: %d, tot_in_bits: %d, tot_out_bits: %d, tot_float32: %d\n", N, tot_bits_in, tot_bits_out, tot_float32);
#endif
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float32}), &output_tensor));

            auto output = output_tensor->flat<float>();

			int i_idx = 0;
            int e_bits = FP_LEN - bround;

			for(int i = 0; i < N; i++) {

                utmpi.f = input(i);
                int acc_bits = 0;
                while (acc_bits < e_bits){
                    utmpi.f = input(i);
                    int in_ptr = i_idx / FP_LEN; // element index
                    int in_off = i_idx % FP_LEN; // bit offset
                    if (in_off == 0) {
                        utmpo.i = 0;
                    } else {
                        utmpo.f = output(in_ptr);
                    }
                    int r_bits = e_bits - acc_bits;
                    int s_container = FP_LEN - in_off;
                    int bits_in = s_container<=r_bits ? s_container : r_bits;
                    int remains_out = FP_LEN - bits_in - in_off;
                    int remains_in = FP_LEN - bits_in - acc_bits;
                    //std::printf("acc: %d, i_idx: %d, in_ptr: %d, in_off: %d, bits_in %d, r_in: %d\n", acc_bits, i_idx, in_ptr, in_off, bits_in, remains_in);
                    unsigned int _and = 0;
                    unsigned int one = 1;
                    for (int j = 0; j < bits_in; j++){
                        _and = _and << 1;
                        _and += one;
                    }
                    _and = _and << remains_in;
                    utmpi.i = utmpi.i & _and;
                    int gap = in_off - acc_bits;
                    if (gap > 0) {
                        utmpi.i = utmpi.i >> gap;
                    } else if (gap < 0) {
                        utmpi.i = utmpi.i << -gap;
                    }
                    utmpo.i += utmpi.i;
                    output(in_ptr) = utmpo.f;
                    acc_bits += bits_in;
                    i_idx += bits_in;
                }
            }

#ifdef __DEBUG__
            std::printf("size: %d\n", output.size());
            for(int i = 0; i < tot_float32; i++){
                utmpo.f = output(i);
                std::printf("0x%X\n", utmpo.i);
            }
#endif
        }
};

REGISTER_KERNEL_BUILDER(Name("BitPack").Device(DEVICE_CPU), BitPackOp);

REGISTER_OP("BitUnpack") .Input("packed: float32") .Input("bits: int32") .Input("nout: int32")
    .Output("unpacked: float32");

class BitUnpackOp : public OpKernel {
    public:
        explicit BitUnpackOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {

            const Tensor& traw = context->input(0);
            auto input = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const int tot_float = nval(0);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float}), &output_tensor));
            auto output = output_tensor->flat<float>();

            BitUnpackLauncher(input.data(), bits(0), nval(0), output.data(), context->eigen_device<Eigen::GpuDevice>());
        }
};

REGISTER_KERNEL_BUILDER(Name("BitUnpack").Device(DEVICE_GPU) .HostMemory("bits") .HostMemory("nout"), BitUnpackOp);
*/

REGISTER_OP("BitPackCpu")
    .Input("input: float32")
    .Input("bits: int32")
    .Output("packed: float32");

class BitPackCpuOp : public OpKernel {
    public:
        explicit BitPackCpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
            start = std::chrono::high_resolution_clock::now();
            const Tensor& traw = context->input(0);
            const Tensor& tbits = context->input(1);
            auto input_eigen = traw.flat<float>();
            auto bits = tbits.flat<int32>();
            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int FP_LEN = BYTE * FP_BYTES;
            const int MANTISSA = 23;
            const int bround = MANTISSA - bits(0);
            int ebits = FP_LEN - bround;
            ebits = bits(0);

            const int tot_float_in = input_eigen.size();
            const int tot_int8_in = tot_float_in * FP_BYTES;
            const int int8_per_float = (ebits + BYTE - 1) / BYTE;
            const int tot_int8_out = tot_float_in * int8_per_float;
            const int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

            //if (sizeof(int) != FP_BYTES) std::printf("WARNING: sizeof int: %lu\n", sizeof(int));
            //if (tot_float_in == tot_float_out) std::printf("WARNING: float in==out\n");
            //std::printf("tot_float_in: %d, tot_int8_in: %d, int8_per_float: %d, tot_int8_out: %d, tot_float_out: %d\n", 
            //        tot_float_in, tot_int8_in, int8_per_float, tot_int8_out, tot_float_out);

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float_out}), &output_tensor));

            auto output_eigen = output_tensor->flat<float>();

            typedef union {
                unsigned char *c; 
                unsigned int *i;
                float *f;
            } _in_unip;
            _in_unip input, output, src, dst;

            input.f = (float *) input_eigen.data();
            output.f = output_eigen.data();

            //end = std::chrono::high_resolution_clock::now();
            //std::cout << std::chrono::duration<double>(end-start).count() << "s----\n";

            int workers = omp_get_max_threads();
            int i;
            #pragma omp parallel for shared(i, input, output) private(src, dst)
            for(i = 0; i < tot_float_in; i++) {
                int skip = FP_BYTES - int8_per_float;
                src.f = input.f + i;
                src.c = src.c + skip;
                dst.c = output.c + (i*int8_per_float);
                int chunk_size = int8_per_float;
                std::memcpy(dst.c, src.c, chunk_size);
            }
            end = std::chrono::high_resolution_clock::now();
            std::cout << 1000000 * std::chrono::duration<double>(end-start).count() << " us\n";
            //std::cout << std::chrono::duration<double>(end-start).count() << "s\n";
        }
};

REGISTER_KERNEL_BUILDER(Name("BitPackCpu").Device(DEVICE_CPU), BitPackCpuOp);

REGISTER_OP("BitUnpackCpu")
    .Input("compressed: float32")
    .Input("bits: int32")
    .Input("nout: int32")
    .Input("shape: int32")
    .Output("unpacked: float32");

class BitUnpackCpuOp : public OpKernel {
    public:
        explicit BitUnpackCpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned char *c;
                unsigned int *i;
                float *f;
            } _in_unip;
            _in_unip srcp, dstp;

            const Tensor& traw = context->input(0);
            auto input_eigen = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int MANTISSA = 23;
            const int FP_LEN = BYTE * FP_BYTES;
            const int bround = MANTISSA - bits(0);
            int ebits = FP_LEN - bround;
            ebits = bits(0);

            const int int8_per_float = (ebits + BYTE -1) / BYTE;
            const int tot_float_out = nval(0);
            const int tot_int8_out = tot_float_out * FP_BYTES;
            const int tot_int8_in = tot_float_out * int8_per_float;
            const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

            const Tensor& sizes = context->input(3);
            const int64 num_dims = sizes.NumElements();
            TensorShape shape;
            auto Svec = sizes.flat<int32>();
            for (int d = 0; d < num_dims; d++) {
                const int32 size = Svec(d);
                shape.AddDim(size);
            }

            const int tot_float = nval(0);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
            //OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float}), &output_tensor));
            auto output_eigen = output_tensor->flat<float>();

            float *input = (float *) input_eigen.data();
            float *output = output_eigen.data();

            int i;
            #pragma omp parallel for shared(i, input, output) private(srcp, dstp)
            for(i = 0; i < tot_float_out; i++) {
                srcp.c = (unsigned char *) input + i * int8_per_float;
                dstp.f = output + i;
                int skip = FP_BYTES - int8_per_float;
                std::memset(dstp.f, 0, skip);
                unsigned char *dst = dstp.c + skip;
                std::memcpy(dst, srcp.c, int8_per_float);
                unsigned int aux = ~0;
                aux <<= FP_LEN - ebits;
                *dstp.i = *dstp.i & aux;
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("BitUnpackCpu").Device(DEVICE_CPU), BitUnpackCpuOp);


REGISTER_OP("BitPackGpu")
    .Input("input: float32")
    .Input("bits: int32")
    .Output("packed: float32");

class BitPackGpuOp : public OpKernel {
    public:
        explicit BitPackGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& traw = context->input(0);
            const Tensor& tbits = context->input(1);
            auto input = traw.flat<float>();
            auto bits = tbits.flat<int32>();

            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int FP_LEN = BYTE * FP_BYTES;
            const int MANTISSA = 23;
            const int bround = MANTISSA - bits(0);
            int ebits = FP_LEN - bround;
            ebits = bits(0);

            const int tot_float_in = input.size();
            const int tot_int8_in = tot_float_in * FP_BYTES;
            const int int8_per_float = (ebits + BYTE - 1) / BYTE;
            const int tot_int8_out = tot_float_in * int8_per_float;
            const int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float_out}), &output_tensor));
            auto output = output_tensor->flat<float>();

            BitPackGpuLauncher(input.data(), bits(0), tot_float_in, output.data(), context->eigen_device<Eigen::GpuDevice>());
        }
};

REGISTER_KERNEL_BUILDER(Name("BitPackGpu").Device(DEVICE_GPU) .HostMemory("bits"), BitPackGpuOp);


REGISTER_OP("BitUnpackGpu")
    .Input("compressed: float32")
    .Input("bits: int32")
    .Input("nout: int32")
    .Input("shape: int32")
    .Output("unpacked: float32");

class BitUnpackGpuOp : public OpKernel {
    public:
        explicit BitUnpackGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {

            const Tensor& traw = context->input(0);
            auto input = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const Tensor& sizes = context->input(3);
            const int64 num_dims = sizes.NumElements();
            TensorShape shape;
            auto Svec = sizes.flat<int32>();
            for (int d = 0; d < num_dims; d++) {
                const int32 size = Svec(d);
                shape.AddDim(size);
            }

            const int tot_float = nval(0);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
            //OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float}), &output_tensor));
            auto output = output_tensor->flat<float>();

            BitUnpackGpuLauncher(input.data(), bits(0), nval(0), output.data(), context->eigen_device<Eigen::GpuDevice>());

        }
};

REGISTER_KERNEL_BUILDER(Name("BitUnpackGpu").Device(DEVICE_GPU) .HostMemory("bits") .HostMemory("nout") .HostMemory("shape"), BitUnpackGpuOp);


REGISTER_OP("BitPackCpuAvx")
    .Input("input: float32")
    .Input("bits: int32")
    .Output("packed: float32");

class BitPackCpuAvxOp : public OpKernel {
    public:
        explicit BitPackCpuAvxOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
            start = std::chrono::high_resolution_clock::now();
            const Tensor& traw = context->input(0);
            const Tensor& tbits = context->input(1);
            auto input_eigen = traw.flat<float>();
            auto bits = tbits.flat<int32>();
            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int FP_LEN = BYTE * FP_BYTES;

            int ebits = bits(0);
            int b;
            if (ebits % BYTE) {
                b = (ebits + BYTE - 1) / BYTE;
                ebits = b * BYTE;
            }
            int dbits = FP_LEN - ebits;


            const int tot_float_in = input_eigen.size();
            const int tot_int8_in = tot_float_in * FP_BYTES;
            const int int8_per_float = (ebits + BYTE - 1) / BYTE;
            const int tot_int8_out = tot_float_in * int8_per_float;
            const int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float_out}), &output_tensor));

            auto output_eigen = output_tensor->flat<float>();

            //end = std::chrono::high_resolution_clock::now();
            //std::cout << std::chrono::duration<double>(end-start).count() << "s----\n";

            typedef union {
                unsigned char *c; 
                unsigned int *i;
                float *f;
            } _in_unip;
            _in_unip conv0, conv1;
            int count = 0;
            int vector_len = 8; /* We are assuming AVX2 here */
            int vector_lane = 4; /* YMM register has two 128-bit lanes */
            int vpos = vector_len * FP_BYTES;
            int iters = (tot_float_in + vector_len - 1) / vector_len;
            //printf("tot_float_in: %d, iters: %d, dbits: %d\n", tot_float_in, iters, dbits);

            int eint32 = int8_per_float * vector_lane / FP_BYTES;
            int dint32 = vector_lane - eint32;
            int dbytes = vector_lane * (FP_BYTES - int8_per_float);
            int ebytes = vector_lane * int8_per_float;
            char e8[vpos];
            int e32[vector_len];

            int top = FP_BYTES * vector_lane - 1;
            int cfloat = 0;
            int boff = 0;

            /* 128-bit lane shuffling */
            for (int j = top; j >= dbytes; j--) {
                int pos = top - boff - cfloat * FP_BYTES;
                e8[j] = e8[j+vpos/2] = pos;
                boff += (boff == int8_per_float - 1) ? (-boff) : 1;
                cfloat += (boff == 0) ? 1 : 0;
            }
            for (int j = 0; j < dbytes; j++) 
                e8[j] = e8[j+vpos/2] = 0x80; /* 0b10000000 */
            __m256i ymm_128 = _mm256_set_epi8(e8[31], e8[30], e8[29], e8[28], e8[27], e8[26], e8[25], e8[24], e8[23], e8[22], e8[21], e8[20], \
                    e8[19], e8[18], e8[17], e8[16], e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
                    e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            /* Cross-lane shuffling (256-bit) */
            int loff = 4;
            cfloat = 0;
            for (int j = vector_len - 1; j >= eint32 * 2; j--) {
                e32[j] = 0;
            }
            for (int j = eint32 * 2 - 1; j >= 0; j--) {
                e32[j] = vector_lane - 1 + loff - cfloat;
                cfloat += (cfloat + 1 == eint32) ? (-cfloat) : 1;
                loff -= (cfloat == 0) ? 4 : 0;
            }
            __m256i ymm_256 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

            for (int j = 0; j < vector_len; j++) {
                if (j >= eint32*2)
                    e32[j] = 0;
                else
                    e32[j] = -1;
            }
            __m256i ymm_256_2 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

            conv1.f = output_eigen.data();
            float* in = (float*) input_eigen.data();
            for (int i = 0; i < iters; i++) {

                conv0.f = in + (i * vector_len);

                __m256i ymm0 = _mm256_loadu_si256((__m256i*) conv0.i);

                __m256i ymm2 = _mm256_shuffle_epi8(ymm0, ymm_128);

                __m256i ymm4 = _mm256_permutevar8x32_epi32(ymm2, ymm_256);

                _mm256_maskstore_epi32((int*) conv1.i, ymm_256_2, ymm4);
                
                conv1.i += eint32 * 2;
            }
            end = std::chrono::high_resolution_clock::now();
            //std::cout << 1000000 * std::chrono::duration<double>(end-start).count() << " us\n";
        }
};

REGISTER_KERNEL_BUILDER(Name("BitPackCpuAvx").Device(DEVICE_CPU), BitPackCpuAvxOp);


REGISTER_OP("BitPackCpuAvxOmp")
    .Input("input: float32")
    .Input("bits: int32")
    .Output("packed: float32");

class BitPackCpuAvxOmpOp : public OpKernel {
    public:
        explicit BitPackCpuAvxOmpOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
            start = std::chrono::high_resolution_clock::now();
            const Tensor& traw = context->input(0);
            const Tensor& tbits = context->input(1);
            //const Tensor& ttog = context->input(2);
            auto input_eigen = traw.flat<float>();
            auto bits = tbits.flat<int32>();
            //auto toggles = ttog.flat<int32>();
            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int FP_LEN = BYTE * FP_BYTES;

            //int toggle = toggles(0);
            int ebits = bits(0);
            int b;
            if (ebits % BYTE) {
                b = (ebits + BYTE - 1) / BYTE;
                ebits = b * BYTE;
            }
            int dbits = FP_LEN - ebits;


            const int tot_float_in = input_eigen.size();
            const int tot_int8_in = tot_float_in * FP_BYTES;
            const int int8_per_float = (ebits + BYTE - 1) / BYTE;
            const int tot_int8_out = tot_float_in * int8_per_float;
            const int tot_float_out = (tot_int8_out + FP_BYTES - 1) / FP_BYTES;

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float_out}), &output_tensor));

            auto output_eigen = output_tensor->flat<float>();

            typedef union {
                unsigned char *c; 
                unsigned int *i;
                float *f;
            } _in_unip;
            _in_unip conv0, conv1;

            int vec_len = 8; /* We are assuming AVX2 here */
            int vec_lane = 4; /* YMM register has two 128-bit lanes */
            int vpos = vec_len * FP_BYTES;
            int omp_workers = omp_get_max_threads();
            int tot_256 = (tot_float_in + vec_len - 1) / vec_len; //number of 256-bit chunks in total
            int thd_256 = (tot_256 + omp_workers - 1) / omp_workers; //number of 256-bit chunks per thread
            int thd_32 = thd_256 * vec_len; //number of 32-bit floats per thread
            int thd_8_out = int8_per_float * thd_32;

            int eint32 = int8_per_float * vec_lane / FP_BYTES;
            int dint32 = vec_lane - eint32;
            int dbytes = vec_lane * (FP_BYTES - int8_per_float);
            int ebytes = vec_lane * int8_per_float;
            char e8[vpos];
            int e32[vec_len];

            int top = FP_BYTES * vec_lane - 1;
            int cfloat = 0;
            int boff = 0;

            /* 128-bit lane shuffling */
            for (int j = top; j >= dbytes; j--) {
                int pos = top - boff - cfloat * FP_BYTES;
                e8[j] = e8[j+vpos/2] = pos;
                boff += (boff == int8_per_float - 1) ? (-boff) : 1;
                cfloat += (boff == 0) ? 1 : 0;
            }
            for (int j = 0; j < dbytes; j++) 
                e8[j] = e8[j+vpos/2] = 0x80; /* 0b10000000 */
            __m256i ymm_128 = _mm256_set_epi8(e8[31], e8[30], e8[29], e8[28], e8[27], e8[26], e8[25], e8[24], e8[23], e8[22], e8[21], e8[20], \
                    e8[19], e8[18], e8[17], e8[16], e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
                    e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            /* Cross-lane shuffling (256-bit) */
            int loff = 4;
            cfloat = 0;
            for (int j = vec_len - 1; j >= eint32 * 2; j--) {
                e32[j] = 0;
            }
            for (int j = eint32 * 2 - 1; j >= 0; j--) {
                e32[j] = vec_lane - 1 + loff - cfloat;
                cfloat += (cfloat + 1 == eint32) ? (-cfloat) : 1;
                loff -= (cfloat == 0) ? 4 : 0;
            }
            __m256i ymm_256 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

            for (int j = 0; j < vec_len; j++) {
                if (j >= eint32*2)
                    e32[j] = 0;
                else
                    e32[j] = -1;
            }
            __m256i ymm_256_2 = _mm256_set_epi32(e32[7], e32[6], e32[5], e32[4], e32[3], e32[2], e32[1], e32[0]);

            float *in = (float*) input_eigen.data();
            float *out = (float*) output_eigen.data();
            #pragma omp parallel for shared(in, out, ymm_128, ymm_256, ymm_256_2) private(conv0, conv1)
            for (int i = 0; i < omp_workers; i++) {
                int off_256 = i * thd_256;
                int iter_256 = (i < omp_workers - 1) ? thd_256 : (tot_256 - off_256);

                conv1.c = (unsigned char*) out + i * thd_8_out;
                conv0.f = in + (i * thd_32);

                for (int j = 0; j < iter_256; j++) {
                    __m256i ymm0 = _mm256_loadu_si256((__m256i*) conv0.i);

                    __m256i ymm2 = _mm256_shuffle_epi8(ymm0, ymm_128);

                    __m256i ymm4 = _mm256_permutevar8x32_epi32(ymm2, ymm_256);

                    _mm256_maskstore_epi32((int*) conv1.i, ymm_256_2, ymm4);
                    
                    conv0.f += vec_len;

                    conv1.i += eint32 * 2;
                }
            }
            end = std::chrono::high_resolution_clock::now();
            //std::cout << 1000000 * std::chrono::duration<double>(end-start).count() << " us\n";
        }
};

REGISTER_KERNEL_BUILDER(Name("BitPackCpuAvxOmp").Device(DEVICE_CPU), BitPackCpuAvxOmpOp);

REGISTER_OP("BitUnpackCpuSse")
    .Input("compressed: float32")
    .Input("bits: int32")
    .Input("nout: int32")
    .Input("shape: int32")
    .Output("unpacked: float32");

class BitUnpackCpuSseOp : public OpKernel {
    public:
        explicit BitUnpackCpuSseOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned char *c;
                unsigned int *i;
                float *f;
            } _in_unip;
            _in_unip srcp, dstp;

            const Tensor& traw = context->input(0);
            auto input_eigen = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int FP_LEN = BYTE * FP_BYTES;
            int ebits = bits(0);
            int b;
            if (ebits % BYTE) {
                b = (ebits + BYTE - 1) / BYTE;
                ebits = b * BYTE;
            }
            int dbits = FP_LEN - ebits;

            const int int8_per_float = (ebits + BYTE -1) / BYTE;
            const int tot_float_out = nval(0);
            const int tot_int8_out = tot_float_out * FP_BYTES;
            const int tot_int8_in = tot_float_out * int8_per_float;
            const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

            const Tensor& sizes = context->input(3);
            const int64 num_dims = sizes.NumElements();
            TensorShape shape;
            auto Svec = sizes.flat<int32>();
            for (int d = 0; d < num_dims; d++) {
                const int32 size = Svec(d);
                shape.AddDim(size);
            }

            const int tot_float = nval(0);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
            auto output_eigen = output_tensor->flat<float>();

            float *input = (float *) input_eigen.data();
            float *output = output_eigen.data();

            const int vec_len = 4;
            const int vpos = vec_len * FP_BYTES;
            char e8[vpos];
            int e32[vec_len];
            for (int i = 0, idx = 0; i < vpos; i++) {
                if ((i % FP_BYTES) >= int8_per_float) {
                    e8[i] = idx; 
                    idx += 1;
                } else {
                    e8[i] = 0x80;
                }
            }
            __m128i ymm_128 = _mm_set_epi8(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], e8[9], e8[8], \
                                           e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            unsigned int aux = ~0;
            aux <<= FP_LEN - bits(0);
            __m128i ymm_and = _mm_set_epi32(aux, aux, aux, aux);

            int iters = (tot_float_out + vec_len - 1) / vec_len;
            srcp.f = input;
            dstp.f = output;

            for (int i = 0; i < iters; i++) {
                __m128i ymm0 = _mm_lddqu_si128((__m128i *) srcp.i);

                __m128i ymm2 = _mm_shuffle_epi8(ymm0, ymm_128);

                __m128i ymm4 = _mm_and_si128(ymm2, ymm_and);

                _mm_store_si128((__m128i*) dstp.f, ymm4);

                srcp.c += vec_len * int8_per_float;

                dstp.i += vec_len;
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("BitUnpackCpuSse").Device(DEVICE_CPU), BitUnpackCpuSseOp);

REGISTER_OP("BitUnpackCpuSseOmp")
    .Input("compressed: float32")
    .Input("bits: int32")
    .Input("nout: int32")
    .Input("shape: int32")
    .Output("unpacked: float32");

class BitUnpackCpuSseOmpOp : public OpKernel {
    public:
        explicit BitUnpackCpuSseOmpOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned char *c;
                unsigned int *i;
                float *f;
            } _in_unip;
            _in_unip srcp, dstp;

            const Tensor& traw = context->input(0);
            auto input_eigen = traw.flat<float>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const int BYTE = 8;
            const int FP_BYTES = 4;
            const int FP_LEN = BYTE * FP_BYTES;
            int ebits = bits(0);
            int b;
            if (ebits % BYTE) {
                b = (ebits + BYTE - 1) / BYTE;
                ebits = b * BYTE;
            }
            int dbits = FP_LEN - ebits;

            const int int8_per_float = (ebits + BYTE -1) / BYTE;
            const int tot_float_out = nval(0);
            const int tot_int8_out = tot_float_out * FP_BYTES;
            const int tot_int8_in = tot_float_out * int8_per_float;
            const int tot_float_in = (tot_int8_in + FP_BYTES - 1) / FP_BYTES;

            const Tensor& sizes = context->input(3);
            const int64 num_dims = sizes.NumElements();
            TensorShape shape;
            auto Svec = sizes.flat<int32>();
            for (int d = 0; d < num_dims; d++) {
                const int32 size = Svec(d);
                shape.AddDim(size);
            }

            const int tot_float = nval(0);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
            auto output_eigen = output_tensor->flat<float>();

            float *input = (float *) input_eigen.data();
            float *output = output_eigen.data();

            const int vec_len = 4;
            const int vpos = vec_len * FP_BYTES;
            char e8[vpos];
            int e32[vec_len];

            int omp_workers = omp_get_max_threads();
            int tot_128 = (tot_float_out + vec_len - 1) / vec_len; //number of 128-bit chunks in total
            int thd_128 = (tot_128 + omp_workers - 1) / omp_workers; //number of 128-bit chunks per thread
            int thd_32 = thd_128 * vec_len; //number of 32-bit floats per thread
            int thd_8_out = FP_BYTES * thd_32;

            for (int i = 0, idx = 0; i < vpos; i++) {
                if ((i % FP_BYTES) >= int8_per_float) {
                    e8[i] = idx; 
                    idx += 1;
                } else {
                    e8[i] = 0x80;
                }
            }
            __m128i ymm_128 = _mm_set_epi8(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], e8[9], e8[8], \
                                           e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            unsigned int aux = ~0;
            aux <<= FP_LEN - bits(0);
            __m128i ymm_and = _mm_set_epi32(aux, aux, aux, aux);

            #pragma omp parallel for shared(input, output, ymm_128, ymm_and) private(srcp, dstp)
            for (int i = 0; i < omp_workers; i++) {
                int off_128 = i * thd_128;
                int iter_128 = (i < omp_workers - 1) ? thd_128 : (tot_128 - off_128);

                dstp.f = output + i * thd_32;
                srcp.c = (unsigned char*) input + i * thd_32 * int8_per_float;

                for (int j = 0; j < iter_128; j++) {
                     __m128i ymm0 = _mm_lddqu_si128((__m128i *) srcp.i);

                    __m128i ymm2 = _mm_shuffle_epi8(ymm0, ymm_128);

                    __m128i ymm4 = _mm_and_si128(ymm2, ymm_and);

                    _mm_store_si128((__m128i*) dstp.f, ymm4);
                   
                    srcp.c += vec_len * int8_per_float;

                    dstp.i += vec_len;
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("BitUnpackCpuSseOmp").Device(DEVICE_CPU), BitUnpackCpuSseOmpOp);
