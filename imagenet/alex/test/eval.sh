#!/usr/bin/env bash

#@job_name = valex
#@output = valex_%J.out
#@error = valex_%J.err
#@initialdir = .
#@total_tasks = 4
#@cpus_per_task = 1
#@gpus_per_node= 1
#@features = k80
#@wall_clock_limit = 00:20:00
#@exclusive

EXE="/home/bsc28/bsc28687/minotauro/ann/tensorflow/imagenet/alex_adaptive/imagenet_eval.py"

python $EXE --checkpoint_dir=/home/bsc28/bsc28687/minotauro/ann/tensorflow/imagenet/alex_adaptive/test/imagenet_train --eval_dir=imagenet_eval --run_once
