#!/usr/bin/env bash

#@job_name = inception
#@output = inception_%J.out
#@error = inception_%J.err
#@initialdir = .
#@total_tasks = 16
#@cpus_per_task = 1
#@gpus_per_node= 4
#@features = k80
#@wall_clock_limit = 30:00:00
#@exclusive

EXE="/home/bsc28/bsc28687/minotauro/ann/tensorflow/imagenet/vgg/imagenet_train.py"

python $EXE --num_gpus=4 --batch_size=128 --train_dir=imagenet_train --data_dir=/gpfs/scratch/bsc28/bsc28687/image-net/ILSVRC/tfrecord0 --max_steps=100000
