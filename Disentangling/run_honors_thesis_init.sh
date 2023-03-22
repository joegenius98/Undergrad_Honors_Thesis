#! /bin/sh

if [ ! -d "train_logs" ]; then
  mkdir train_logs
fi

# To load from the last checkpoint, specify --ckpt_name last

nice python main.py --train True --seed 1 --cuda False --gpu 0 --max_iter 1.5e6 --batch_size 22 \
    --lr 1e-4 --beta1 0.9 --beta2 0.999 --z_dim 10 --model L --objective L \
    --beta 4 --beta_TC 6 --C_tc_start 0 --C_tc_max 5 --C_tc_step_val 0.02 --lambda_tc 1 \
    --num_sim_factors 1 --augment_factor 1 \
    --dset_dir ./data --dataset dsprites --image_size 64 --num_workers 4 \
    --viz_on True --viz_name Honors_Thesis_First_Run --viz_port 8097 --gather_step 10000 \
    --save_step 10000 --save_output True --output_dir ./outputs --limit 3 \
    --ckpt_dir ./checkpoints \
