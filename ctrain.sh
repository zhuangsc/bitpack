#!/usr/bin/env bash

#@job_name = alex_64
#@output = alex_64_1_24_6_%J.out
#@error = alex_64_1_24_6_%J.err
#@initialdir = .
#@total_tasks = 16
#@cpus_per_task = 1
#@gpus_per_node= 4
#@wall_clock_limit = 08:00:00
#@exclusive

EXE="/path/to/imagenet/alex/imagenet_train.py"

PROFILE=False
DATA_DIR="/path/to/Imagenet/TFRecords"
GPUS=4
STEPS=48060
BATCH=64

BITPACK=0
DIGITS=16
REL_RES=8E-5
INTERVAL=10
STRIDE=0
INITIAL_LEARNING_RATE=0.01

SAVER_DIR=""
TRAIN_DIR="alex_1_24_6"

### CODE
DIR=alex_train_${PROFILE}_${STEPS}_${GPUS}_${BATCH}_${BITPACK}_${DIGITS}_${REL_RES}_${INTERVAL}_${STRIDE}
echo "${DIR} ${SAVER_DIR} ${TRAIN_DIR}"
if [ ! -e ${DIR} ]
then
    mkdir ${DIR}
fi
cd ${DIR}

OMP_NUM_THREADS=16 python -u $EXE --num_gpus=$GPUS --batch_size=$BATCH  --initial_learning_rate=$INITIAL_LEARNING_RATE \
                               --train_dir=$TRAIN_DIR --data_dir=$DATA_DIR \
                               --max_steps=$STEPS --use_bitpack=$BITPACK --profile=$PROFILE --digits=$DIGITS \
                               --rel_res=$REL_RES --interval=$INTERVAL --stride=$STRIDE \
                               --pretrained_model_checkpoint_path=${SAVER_DIR}
