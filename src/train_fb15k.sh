#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--embedding_dim 100 \
--learning_rate 0.003 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 20