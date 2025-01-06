#!/bin/bash

data_n='ml-1m'
lr=5e-5 # ml-1m: 5e-5, game: 5e-5, anime: 5e-5
wd=0.001 # ml-1m: 0.001, game: 0.1 , anime: 0.01
lam=1 # ml-1m: 1, game: 1e-2 , anime: 1
w_max=1.6 # ml-1m: 1.6, game: 1.3, anime: 2
w_min=1 # ml-1m: 1, game: 1, anime: 0.8
drop=0.3 # ml-1m: 0.3, game: 0.3 , anime: 0.1

schedule=linear-var

step=5 # ml-1m: 5, game: 15, anime: 15
scale=1 # ml-1m: 1, game: 1e-4, anime: 1e-4
bs=0.005 # ml-1m: 0.005, game: 0.0005, anime: 0.0005
be=0.05 # ml-1m: 0.05, game: 0.005, anime: 0.005
drop_div=0.3 # ml-1m: 0.3, game: 0.3, anime: 0.5
guide=-0.5 # ml-1m: -0.5, game: -0.7, anime: -0.5


python inference.py \
    --step ${step} \
    --beta_start ${bs} \
    --beta_end ${be} \
    --noise_scale ${scale} \
    --noise_schedule ${schedule} \
    --dataset_name ${data_n} \
    --w_max ${w_max} \
    --w_min ${w_min} \
    --snr \
    --lamda ${lam} \
    --drop_div ${drop_div} \
    --dropout ${drop} \
    --guide_w ${guide}