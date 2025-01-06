#!/bin/bash

data_n=ml-1m

lr=5e-5 # ml-1m: 5e-5, game: 5e-5, anime: 5e-5
wd=0.01 # ml-1m: 0.01, game: 0.1, anime: 0.1
lam=1 # ml-1m: 1, game: 1e-2, anime: 1e-4
w_min=0.8 # ml-1m: 0.8, game: 1.0
w_max=1.3 # ml-1m: 1.3, game: 1.3
drop=0.5 # ml-1m: 0.5, game: 0.5, anime: 0.5
drop_div=0.3 # ml-1m: 0.3, game: 0.3, anime: 0.1
schedule=linear-var

step=5 # ml-1m: 5 game: 15, anime: 15
scale=1e-4 # ml-1m: 1e-4 game: 1, anime: 1e-4
bs=0.005 # ml-1m: 0.005, game: 0.005, anime: 0.005
be=0.01 # ml-1m: 0.01, game: 0.05, anime: 0.05
guide=7


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