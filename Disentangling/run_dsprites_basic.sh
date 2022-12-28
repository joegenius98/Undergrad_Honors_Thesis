#! /bin/sh

nice python3 main.py --train True --seed 1 --cuda True --gpu 0 --max_iter 1.5e6 --batch_size 64 \
    --limit 3 --is_PID False --z_dim 10 --beta 4 --objective H --model H \
    --lr 1e-4 --beta1 0.9 --beta2 0.999 --dset_dir ./data --dataset dsprites --image_size 64 \
    --num_workers 4 --viz_on True --viz_name Higgins_Basic_for_Joseph --viz_port 8097 --save_output True \
    --output_dir ./outputs --gather_step 10000 --save_step 10000 --ckpt_dir ./checkpoints 
