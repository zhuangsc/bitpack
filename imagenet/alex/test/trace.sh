#!/usr/bin/env bash

#@job_name = alex+1
#@output = alex+1_%J.out
#@error = alex+1_%J.err
#@initialdir = .
#@total_tasks = 16
#@cpus_per_task = 1
#@gpus_per_node= 4
#@features = k80
#@wall_clock_limit = 00:20:00
#@exclusive

EXE="/home/bsc28/bsc28687/minotauro/ann/tensorflow/imagenet/alex/imagenet_train.py"

PROFILE=False
DATA_DIR="/gpfs/scratch/bsc28/bsc28687/image-net/ILSVRC/tfrecord0"
TRAIN_DIR=imagenet_train
STEPS=500
GPUS=4
BATCH=256

DIGITS=16
REL_RES=1E-4
INTERVAL=10
STRIDE=0

python $EXE --num_gpus=$GPUS --batch_size=$BATCH --train_dir=$TRAIN_DIR --data_dir=$DATA_DIR --max_steps=$STEPS --use_bitpack=0 --profile=$PROFILE \
            --digits=$DIGITS --rel_res=$REL_RES --interval=$INTERVAL --stride=$STRIDE

rm $TRAIN_DIR/model*
mv $TRAIN_DIR ${TRAIN_DIR}_0


OMP_NUM_THREADS=8 python $EXE --num_gpus=$GPUS --batch_size=$BATCH --train_dir=$TRAIN_DIR --data_dir=$DATA_DIR --max_steps=$STEPS --use_bitpack=1 --profile=$PROFILE \
            --digits=$DIGITS --rel_res=$REL_RES --interval=$INTERVAL --stride=$STRIDE

rm $TRAIN_DIR/model*
mv $TRAIN_DIR ${TRAIN_DIR}_1

rm *.dat
