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
#include "vec128int.h"
//#include <immintrin.h>
#include <pmmintrin.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

void BitUnpackLauncher(const float* input, const int bits, const int nval, float* output, const GPUDevice& d);
void BitUnpackGpuLauncher(const float* input, const int bits, const int nval, float* output, const GPUDevice& d);
void BitPackGpuLauncher(const float* input, const int bits, const int nval_in, float* output, const GPUDevice& d);

/*
 * Swap a[0...pos] to the end of the array
 */
void swap(char *a, int len, int pos)
{
    int l = 0;
    for (int i = len-pos; i < len; i++, l++) {
        int tmp = a[i];
        a[i] = a[l];
        a[l] = tmp;
    }
}

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
            #pragma omp parallel for shared(input, output) private(src, dst)
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
            #pragma omp parallel for shared(input, output) private(srcp, dstp)
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
            int iters = (tot_float_in + vector_lane - 1) / vector_lane;

            int eint32 = int8_per_float * vector_lane / FP_BYTES; //Output bytes
            int dint32 = vector_lane - eint32;
            int dbytes = vector_lane * (FP_BYTES - int8_per_float);
            int ebytes = vector_lane * int8_per_float;
            char e8[vpos];

            int top = FP_BYTES * vector_lane - 1;
            int cfloat = 0;
            int boff = 0;

            for (int j = top; j >= dbytes; j--) {
                int pos = top - boff - cfloat * FP_BYTES;
                e8[j] = e8[j+vpos/2] = pos;
                boff += (boff == int8_per_float - 1) ? (-boff) : 1;
                cfloat += (boff == 0) ? 1 : 0;
            }
            for (int j = 0; j < dbytes; j++) 
                e8[j] = e8[j+vpos/2] = 0x80; /* 0b10000000 */

            swap(e8, 16, ebytes);

            __m128i ymm_128 = vec_set16sb(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
                    e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            for (int j = 0; j < vector_lane*FP_BYTES; j++) {
                if (j < vector_lane*FP_BYTES-ebytes)
                    e8[j] = 0;
                else
                    e8[j] = 0x80;
            }

            swap(e8, 16, ebytes);

            __m128i ymm_128_2 = vec_set16sb(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
                    e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            conv1.f = output_eigen.data();
            float* in = (float*) input_eigen.data();
            for (int i = 0; i < iters; i++) {
                conv0.f = in + (i * vector_lane);

                __m128i ymm0 = _mm_lddqu_si128((__m128i *) conv0.i);

                __m128i ymm2 = vec_permute16sb(ymm0, ymm_128);

                vec_xst_len(ymm2, conv1.c, ebytes);

                conv1.i += eint32;
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
            int vector_len = 8; /* We are assuming AVX2 here */
            int vector_lane = 4; /* YMM register has two 128-bit lanes */
            int vpos = vector_len * FP_BYTES;
            //int iters = (tot_float_in + vector_lane - 1) / vector_lane;
            int omp_workers = omp_get_max_threads();
            int tot_128 = (tot_float_in + vector_lane - 1) / vector_lane; //number of 128-bit chunks in total
            int thd_128 = (tot_128 + omp_workers - 1) / omp_workers; //number of 128-bit chunks per thread
            int thd_32 = thd_128 * 4; //number of 32-bit floats per thread
            int thd_8_out = int8_per_float * thd_32;

            int eint32 = int8_per_float * vector_lane / FP_BYTES; //Output bytes
            int dint32 = vector_lane - eint32;
            int dbytes = vector_lane * (FP_BYTES - int8_per_float);
            int ebytes = vector_lane * int8_per_float;
            char e8[vpos];

            int top = FP_BYTES * vector_lane - 1;
            int cfloat = 0;
            int boff = 0;

            for (int j = top; j >= dbytes; j--) {
                int pos = top - boff - cfloat * FP_BYTES;
                e8[j] = e8[j+vpos/2] = pos;
                boff += (boff == int8_per_float - 1) ? (-boff) : 1;
                cfloat += (boff == 0) ? 1 : 0;
            }
            for (int j = 0; j < dbytes; j++) 
                e8[j] = e8[j+vpos/2] = 0x80; /* 0b10000000 */

            swap(e8, 16, ebytes);

            __m128i ymm_128 = vec_set16sb(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
                    e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            for (int j = 0; j < vector_lane*FP_BYTES; j++) {
                if (j < vector_lane*FP_BYTES-ebytes)
                    e8[j] = 0;
                else
                    e8[j] = 0x80;
            }

            swap(e8, 16, ebytes);

            __m128i ymm_128_2 = vec_set16sb(e8[15], e8[14], e8[13], e8[12], e8[11], e8[10], \
                    e8[9], e8[8], e8[7], e8[6], e8[5], e8[4], e8[3], e8[2], e8[1], e8[0]);

            float *in = (float*) input_eigen.data();
            float *out = (float*) output_eigen.data();
            #pragma omp parallel for shared(in, out, ymm_128, ymm_128_2, ebytes) private(conv0, conv1)
            for (int i = 0; i < omp_workers; i++) {
                int off_128 = i * thd_128;
                int iter_128 = (i < omp_workers - 1) ? thd_128 : (tot_128 - off_128);

                conv1.c = (unsigned char*) out + i * thd_8_out;
                conv0.f = in + (i * thd_32);

                for (int j = 0; j < iter_128; j++) {
                    __m128i ymm0 = _mm_lddqu_si128((__m128i *) conv0.i);

                    __m128i ymm2 = vec_permute16sb(ymm0, ymm_128);

                    vec_xst_len(ymm2, conv1.c, ebytes);
                    
                    conv0.f += vector_lane;

                    conv1.i += eint32;
                }
            }

            end = std::chrono::high_resolution_clock::now();
            //std::cout << 1000000 * std::chrono::duration<double>(end-start).count() << " us\n";
        }
};

REGISTER_KERNEL_BUILDER(Name("BitPackCpuAvxOmp").Device(DEVICE_CPU), BitPackCpuAvxOmpOp);
