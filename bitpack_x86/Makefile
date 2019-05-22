CC = gcc
CPP = g++ 
NVCC = nvcc

#PREFIX = /home/bsc28/bsc28687/minotauro/ann/tensorflow/bit_packer/float32
PREFIX = ./float32
SO = bit_packer.so
OBJ_GPU = bit_packer.cu.o
SRC_GPU = $(PREFIX)/bit_packer.cu.cc
SRC_CPU = $(PREFIX)/bit_packer.cc

$(SO) : $(SRC_CPU) $(OBJ_GPU)
	$(CPP) -std=c++11 -Wall -shared -o $@ $^ -I $(TF_INC) -fPIC -L/opt/cuda/8.0/lib64 -lcudart -fopenmp -mavx2 -D __OPENMP__

$(OBJ_GPU) : $(SRC_GPU)
	$(NVCC) -std=c++11 -c -o $@ $^ -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

#%.so : %.cc
#	$(CPP) -shared -o $@ $< 


.PHONY: clean

clean:
	rm -f *.so *.o

#-D_GLIBCXX_USE_CXX11_ABI=0 
#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#nvcc -std=c++11 -c -o bit_compressor.cu.o bit_compressor.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 -shared -o bit_compressor.so bit_compressor.cc bit_compressor.cu.o -I $TF_INC -fPIC -L/opt/cuda/8.0/lib64 -lcudart
