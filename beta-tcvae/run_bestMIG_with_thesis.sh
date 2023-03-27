#!/bin/sh

nice python vae_quant.py --dataset shapes --beta 8 --tcvae --visdom --mss --exclude-mutinfo --num-epochs 100 \
 --log_freq 1200 --use_augment_dataloader --augment_factor 50 --num_sim_factors 2 --save bestMIG_with_thesis_aug_50
