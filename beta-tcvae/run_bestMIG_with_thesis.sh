#!/bin/sh

nice python vae_quant.py --dataset shapes --beta 8 --tcvae --exclude-mutinfo --visdom --visdom_port 4500 --mss --num-epochs 100 \
 --log_freq 1200 --use_augment_dataloader --augment_factor 100 --num_sim_factors 1 --save k_eq_1
