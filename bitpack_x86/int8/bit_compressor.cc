#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

#include <cstdio>
#include <chrono>
#include <iostream>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

void BitDecompressLauncher(const signed char* input, const int bits, const int nval, float* output, const GPUDevice& d);

REGISTER_OP("BitCompress")
    .Input("input: float32")
    .Input("bits: int32")
    .Output("compressed: int8");

class BitCompressOp : public OpKernel {
    public:
        explicit BitCompressOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned int i; //Logical shift requires unsigned type
                float f;
            } _in_uni;
            _in_uni utmp, utmp1;

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
            int tot_int8 = (tot_bits_out+8-1)/8;
            if (sizeof(int) != 4)
                std::printf("WARNING!!! sizeof int: %lu\n", sizeof(int));
#ifdef __DEBUG__
                std::printf("N: %d, tot_in_bits: %d, tot_out_bits: %d, tot_int8: %d\n", N, tot_bits_in, tot_bits_out, tot_int8);
#endif
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_int8}), &output_tensor));

            auto output = output_tensor->flat<int8>();

			int i_idx = 0;
            // Fill each element of the output tensor
			for(int i = 0; i < tot_int8; i++) {
                //int skip = 0;
                int in_ptr = i_idx / FP_LEN; // element index
                int in_off = i_idx % FP_LEN; // bit offset
                int in_len = FP_LEN - (in_off + bround); //effecive bits from the current input value
                if (in_len <= 0) {
                    in_ptr += 1;
                    in_off = 0;
                    in_len = BYTE;
                    i_idx = FP_LEN * in_ptr;
                } else {
                    in_len = in_len >= BYTE ? BYTE : in_len;
                }

                int next_ptr = in_ptr;
                int next_bits = 0;

                if (in_len < BYTE && tot_bits_in-i_idx >= BYTE && i+1 < tot_int8) {
                    next_ptr += 1;
                    next_bits = BYTE - in_len;
                }

#ifdef __DEBUG__
                std::printf("i: %d in_ptr: %d in_off: %d in_len: %d next_bits: %d\n", i, in_ptr, in_off, in_len, next_bits);
#endif
                //int itmp = *(int*)&input(in_ptr);
                utmp.f = input(in_ptr);
#ifdef __DEBUG__
                std::printf("utmp.i: 0x%X, utmp.f: %E\n", utmp.i, utmp.f);
#endif
                if (in_ptr == next_ptr) {
                    char tmp = (utmp.i << in_off >> (FP_LEN - in_len)); 
                    output(i) = tmp;
                    i_idx += BYTE;
#ifdef __DEBUG__
                    std::printf("%d %d\n", in_off, FP_LEN-in_len);
                    std::printf("%X %X\n", utmp.i<<in_off, utmp.i<<in_off>>(FP_LEN-in_len));
                    std::printf("utmp.i: 0x%X tmp: 0x%hhX\n", utmp.i, tmp);
#endif
                } else {
                    char tmp = (utmp.i >> (FP_LEN-in_off-in_len) << next_bits);
                    utmp1.f = input(next_ptr);
                    char tmp1 = (utmp1.i >> (FP_LEN-next_bits));
                    output(i) = tmp + tmp1;
                    i_idx = next_ptr * FP_LEN + next_bits;
#ifdef __DEBUG__
                    std::printf("utmp.i: 0x%X utmp1.i: 0x%X tmp: 0x%hhX tmp1: 0x%hhX sum(tmp): 0x%hhX\n", utmp.i, utmp1.i, tmp, tmp1, tmp+tmp1);
#endif
                }
			}
#ifdef __DEBUG__
            for(int i = 0; i < tot_int8; i++){
                std::printf("%hhX\n", output(i));
            }
#endif
        }
};

REGISTER_KERNEL_BUILDER(Name("BitCompress").Device(DEVICE_CPU), BitCompressOp);

REGISTER_OP("BitDecompress")
    .Input("compressed: int8")
    .Input("bits: int32")
    .Input("nout: int32")
    .Output("decompressed: float32");

/*
class BitDecompressOp : public OpKernel {
    public:
        explicit BitDecompressOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned int i;
                float f;
            } _in_uni;
            _in_uni utmp, out;

            const Tensor& traw = context->input(0);
            auto input = traw.flat<int8>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const int BYTE = 8;
            const int FP_LEN = BYTE * 4;
            const int MANTISSA = 23;

            const int bround = MANTISSA - bits(0);
            const int tot_float = nval(0);

            //std::printf("---------------------------------------------------------\n");
            if (sizeof(int) != 4)
                std::printf("WARNING!!! sizeof int: %lu\n", sizeof(int));
			//std::printf("N: %d, tot_in_bits: %d, tot_out_bits: %d, tot_int8: %d\n", N, tot_bits_in, tot_bits_out, tot_int8);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float}), &output_tensor));

            auto output = output_tensor->flat<float>();

			int i_idx = 0;
            // Fill each element of the output tensor
			for(int i = 0; i < tot_float; i++) {
                int e_bits = FP_LEN - bround;
                int acc_bits = 0;
                out.i = 0;
                while (acc_bits < FP_LEN) {
                    int in_ptr = i_idx / BYTE; 
                    int in_off = i_idx % BYTE;
                    int len = e_bits - acc_bits;
                    utmp.i = input(in_ptr);
#ifdef __DEBUG__
                    std::printf("in_ptr: %d, in_off: %d\n", in_ptr, in_off);
                    std::printf("utmp.i: 0x%X\n", utmp.i);
#endif
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
#ifdef __DEBUG__
                    std::printf("0x%X 0x%X 0x%X\n", input(in_ptr), utmp.i, out.i);
                    std::printf("%d %d %d %d %d\n", bits, acc_bits, in_ptr, in_off, shifts);
#endif
                }
#ifdef __DEBUG__
                std::printf("out.i: 0x%X, out.f: %E\n", out.i, out.f);
#endif
                output(i) = out.f;
			}
        }
};

REGISTER_KERNEL_BUILDER(Name("BitDecompress").Device(DEVICE_CPU), BitDecompressOp);
*/

class BitDecompressOp : public OpKernel {
    public:
        explicit BitDecompressOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            typedef union {
                unsigned int i;
                float f;
            } _in_uni;
            _in_uni utmp, out;

            const Tensor& traw = context->input(0);
            auto input = traw.flat<int8>();
            const Tensor& tbits = context->input(1);
            auto bits = tbits.flat<int32>();
            const Tensor& tnval = context->input(2);
            auto nval = tnval.flat<int32>();

            const int tot_float = nval(0);
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({tot_float}), &output_tensor));
            auto output = output_tensor->flat<float>();

            BitDecompressLauncher(input.data(), bits(0), nval(0), output.data(), context->eigen_device<Eigen::GpuDevice>());
        }
};

REGISTER_KERNEL_BUILDER(Name("BitDecompress").Device(DEVICE_GPU) .HostMemory("bits") .HostMemory("nout"), BitDecompressOp);
