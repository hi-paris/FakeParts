#!/bin/bash
export PYTHONPATH=$(pwd)
python train.py \
  --exp_name train_testset_2 \
  datasets testset_2 datasets_test testset_2 batch_size 16 nepoch 20 num_workers 2
