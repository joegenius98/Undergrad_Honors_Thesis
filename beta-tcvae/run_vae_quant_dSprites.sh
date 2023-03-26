#!/bin/sh

nice python vae_quant.py --dataset shapes --beta 8 --tcvae --visdom --mss --exclude-mutinfo --num-epochs 100
