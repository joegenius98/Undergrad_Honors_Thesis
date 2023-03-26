#!/bin/sh

nice python vae_quant.py --dataset shapes --beta 8 --tcvae --visdom --conv --mss --exclude-mutinfo
