#!/bin/sh
# beta is set to 6 to gradually increase it the whole time
nice python vae_quant.py --dataset shapes --beta 6 \
 --tcvae --visdom --conv --mss --exclude-mutinfo --num-epochs 100 \
 --log_freq 1200 --beta-anneal --lambda-anneal
