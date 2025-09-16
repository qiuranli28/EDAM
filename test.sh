#!/bin/bash

DATASET="vcclothes"
TESTSET="prcc"
DATASET_PATH="/root/Documents/dataset/reid/data"
EXP_DIR="./exp"

python main.py --dataset $DATASET --testset $TESTSET --data-dir $DATASET_PATH --exp-dir $EXP_DIR --evaluate
