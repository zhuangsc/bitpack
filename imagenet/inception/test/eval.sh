#!/usr/bin/env bash

#@job_name = vinception
#@output = vinception_%J.out
#@error = vinception_%J.err
#@initialdir = .
#@total_tasks = 16
#@cpus_per_task = 1
#@gpus_per_node= 4
#@features = k80
#@wall_clock_limit = 25:00:00
#@exclusive

EXE="/home/bsc28/bsc28687/minotauro/ann/tensorflow/imagenet/inception/imagenet_eval.py"

python $EXE --checkpoint_dir=/home/bsc28/bsc28687/minotauro/ann/tensorflow/imagenet/inception/test/imagenet_train --eval_dir=imagenet_eval --run_once
