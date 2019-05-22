## Power9 makefile

CC = xlc_r
CPP = xlc++_r
NVCC = nvcc

PREFIX = ./float32
SO = bit_packer.so
OBJ_GPU = bit_packer.cu.o
SRC_GPU = $(PREFIX)/bit_packer.cu.cc
SRC_CPU = $(PREFIX)/bit_packer.cc
POWER_FLAGS=-DNDEBUG -O3 -qsmp=omp -qipa -qnostrict -qhot -qmaxmem=-1 -qthreaded -qalign=linuxppc -qsimd -qaltivec -q64 -qarch=pwr9 -qtune=pwr9 -qsmp=omp -qipa -qnostrict
GCC_FLAGS=-DNDEBUG -mabi=altivec -fopenmp -maltivec -mcpu=power9 
#TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=-I/usr/local -I./include
#TF_INC=$(shell python -c "import tensorflow; print(tensorflow.sysconfig.get_include())")
#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )                                                                    
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )                                                                         
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

$(SO) : $(SRC_CPU) $(OBJ_GPU)
	#$(CPP) -std=c++11 -shared $(POWER_FLAGS) -o $@ $^ -I./float32 -fPIC -lcudart -D __OPENMP__ -I$(TF_INC) ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=1
	#$(CPP) -std=c++11 -shared $(POWER_FLAGS) -o $@ $^ -I$(TF_INC) -I./float32 -L${TF_LIB} -ltensorflow_framework -fPIC -lcudart -D __OPENMP__ #-D_GLIBCXX_USE_CXX11_ABI=0
	$(CPP) -std=c++11 -shared $(POWER_FLAGS) -o $@ $^ -I./float32 -fPIC -lcudart -D __OPENMP__ $(TF_INC) $(TF_CFLAGS) $(TF_LFLAGS) -Wl,-rpath,/apps/PYTHON/3.6.5_ML/lib/python3.6/site-packages/tensorflow

$(OBJ_GPU) : $(SRC_GPU)
	#$(NVCC) -std=c++11 -c $^ ${TF_CFLAGS[@]} -x cu -Xcompiler -fPIC -o $@ -DGOOGLE_CUDA=1 --expt-relaxed-constexpr #-D_GLIBCXX_USE_CXX11_ABI=0
	#$(NVCC) -std=c++11 -c $^ -I$(TF_INC) -x cu -Xcompiler -fPIC -o $@ -DGOOGLE_CUDA=1 --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=1
	$(NVCC) -std=c++11 -DNDEBUG -c $^ $(TF_INC) $(TF_CFLAGS) $(TF_LFLAGS) -Xcompiler \"-Wl,-rpath,/apps/PYTHON/3.6.5_ML/lib/python3.6/site-packages/tensorflow\" -x cu -Xcompiler -fPIC -o $@ -DGOOGLE_CUDA=1 --expt-relaxed-constexpr

.PHONY: clean

clean:
	rm -f *.so *.o

## IBM xlc 
#xlc_r -O3 -qhot  -qmaxmem=-1 -qthreaded -qalign=linuxppc -qthreaded -qsimd -qaltivec -q64 -qarch=pwr9 -qtune=pwr9 -o avx avx128_ibm2.c
#-D_GLIBCXX_USE_CXX11_ABI=0
