#!/usr/bin/env bash
env_name=$1

for random_seed in 1234 2314 2345 1235; do
    python main.py -c configs/algos/slbo.yml configs/envs/${env_name}.yml -s log_dir=experiments/${env_name}_${random_seed} seed=${random_seed}
done
