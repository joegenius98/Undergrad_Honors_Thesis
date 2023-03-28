#!/bin/sh

nice python vae_quant.py --dataset shapes --beta 8 --tcvae --exclude-mutinfo --mss --visdom --visdom_port 4500 --num-epochs 100 \
 --log_freq 1200  --save reproduce_results
