#! /bin/sh

nice python main.py --seed 1 \
 --dataset dsprites --num_workers 4 --batch_size 64 \
 --output_save True --viz_on True --viz_port 4520 \
 --viz_ll_iter 1000 --viz_la_iter 5000 --viz_ra_iter 50000 --viz_ta_iter 50000 \
 --max_iter 7e5 --print_iter 5000 \
 --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
 --use_augment_dataloader --num_sim_factors 3 --use_sort_strategy --augment_choice 2 --augment_factor 5 \
 --name trans_sort_k3_af5_seed1 --z_dim 10 --gamma 10 --ckpt_load last --ckpt_save_iter 175_000

nice python main.py --seed 2 \
 --dataset dsprites --num_workers 4 --batch_size 64 \
 --output_save True --viz_on True --viz_port 4520 \
 --viz_ll_iter 1000 --viz_la_iter 5000 --viz_ra_iter 50000 --viz_ta_iter 50000 \
 --max_iter 7e5 --print_iter 5000 \
 --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
 --use_augment_dataloader --num_sim_factors 3 --use_sort_strategy --augment_choice 2 --augment_factor 5 \
 --name trans_sort_k3_af5_seed2 --z_dim 10 --gamma 10 --ckpt_load last --ckpt_save_iter 175_000

nice python main.py --seed 3 \
 --dataset dsprites --num_workers 4 --batch_size 64 \
 --output_save True --viz_on True --viz_port 4520 \
 --viz_ll_iter 1000 --viz_la_iter 5000 --viz_ra_iter 50000 --viz_ta_iter 50000 \
 --max_iter 7e5 --print_iter 5000 \
 --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
 --use_augment_dataloader --num_sim_factors 3 --use_sort_strategy --augment_choice 2 --augment_factor 5 \
 --name trans_sort_k3_af5_seed3 --z_dim 10 --gamma 10 --ckpt_load last --ckpt_save_iter 175_000

nice python main.py --seed 4 \
 --dataset dsprites --num_workers 4 --batch_size 64 \
 --output_save True --viz_on True --viz_port 4520 \
 --viz_ll_iter 1000 --viz_la_iter 5000 --viz_ra_iter 50000 --viz_ta_iter 50000 \
 --max_iter 7e5 --print_iter 5000 \
 --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
 --use_augment_dataloader --num_sim_factors 3 --use_sort_strategy --augment_choice 2 --augment_factor 5 \
 --name trans_sort_k3_af5_seed4 --z_dim 10 --gamma 10 --ckpt_load last --ckpt_save_iter 175_000

nice python main.py --seed 5 \
 --dataset dsprites --num_workers 4 --batch_size 64 \
 --output_save True --viz_on True --viz_port 4520 \
 --viz_ll_iter 1000 --viz_la_iter 5000 --viz_ra_iter 50000 --viz_ta_iter 50000 \
 --max_iter 7e5 --print_iter 5000 \
 --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
 --use_augment_dataloader --num_sim_factors 3 --use_sort_strategy --augment_choice 2 --augment_factor 5 \
 --name trans_sort_k3_af5_seed5 --z_dim 10 --gamma 10 --ckpt_load last --ckpt_save_iter 175_000