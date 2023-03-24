#! /bin/sh

nice python main.py --train True --seed 1 --cuda True --gpu 0 --max_iter 1.5e6 --batch_size 66 \
    --lr 1e-4 --beta1 0.9 --beta2 0.999 --z_dim 10 --model L --objective H \
    --beta 100 --dset_dir ./data --dataset dsprites --image_size 64 --num_workers 4 \
    --viz_on True --viz_name Honors_Thesis_Pure_BetaVAE --viz_port 8097 --gather_step 10000 \
    --save_step 10000 --save_output True --output_dir ./outputs --limit 3 \
    --ckpt_dir ./checkpoints \
