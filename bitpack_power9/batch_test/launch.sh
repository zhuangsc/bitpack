#!/usr/bin/env bash

#@job_name = omp_16
#@output = omp_16_%J.out
#@error = omp_16_%J.err
#@initialdir = .
#@total_tasks = 16
#@cpus_per_task = 1
#@gpus_per_node= 4
#@features = k80
#@wall_clock_limit = 00:05:00
#@exclusive

EXE="/home/bsc28/bsc28687/minotauro/ann/tensorflow/bit_packer/test_gpu.py"
THREADS=16

export OMP_NUM_THREADS=$THREADS
echo $OMP_NUM_THREADS
$EXE
