#! /bin/bash

port=$1
sh check_port.sh $port
exit_status=$?

if [ $exit_status -eq 1 ]
then
  exit 1
fi

gpu=$2
if [ -z $gpu ]
then
  echo "Please input GPU id (accr. to nvidia-smi ordering)"
  exit 1
fi

dset="dsprites"
k=1
af=5
aug="sort_rot"
aug_i=1

for seed in {1..5}
do
    nice python main.py --seed $seed \
    --dataset $dset --num_workers 4 --batch_size 64 --gpu $gpu \
    --output_save True --viz_on True --viz_port $port \
    --viz_ll_iter 1000 --viz_la_iter 5000 --viz_ra_iter 50000 --viz_ta_iter 50000 \
    --max_iter 7e5 --print_iter 5000 \
    --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
    --use_augment_dataloader --use_sort_strategy --augment_choice $aug_i --num_sim_factors $k --augment_factor $af \
    --name ${aug}_k${k}_af${af}_seed${seed} --z_dim 10 --gamma 10 --ckpt_load last --ckpt_save_iter 175_000
done
