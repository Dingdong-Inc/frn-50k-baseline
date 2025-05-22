#!/bin/bash

# Define parameters
TOTAL_SEQ_LEN=90
SEQ_LEN=62
MOVING_AVG=28
PRED_LEN=7
ENC_IN=7
MODEL_NAME="DLinear"


# Run
python -u main.py \
    --is_training 1 \
    --train_only True \
    --model "$MODEL_NAME" \
    --scale True \
    --loss 'mae' \
    --features "MS" \
    --data "censored_dataset" \
    --target "sale_amount" \
    --revin False \
    --total_seq_len "$TOTAL_SEQ_LEN" \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --moving_avg "$MOVING_AVG" \
    --enc_in "$ENC_IN" \
    --des "original_l1" \
    --do_predict \
    --itr 1 \
    --batch_size 1024 \
    --train_epochs 6 \
    --num_workers 16 \
    --learning_rate 0.001
