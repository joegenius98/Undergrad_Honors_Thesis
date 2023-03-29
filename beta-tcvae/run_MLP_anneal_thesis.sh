#!/bin/sh
# beta is set to 6 to gradually increase it the whole time
nice python vae_quant.py --dataset shapes --beta 6 \
 --tcvae --exclude-mutinfo --visdom --visdom_port 4521 --mss --num-epochs 100 \
 --log_freq 1200 \
 --beta-anneal --lambda-anneal --use_augment_dataloader --augment_factor 50 --num_sim_factors 1 \
 --save MLP_anneal_thesis_beta1_mutinfo
