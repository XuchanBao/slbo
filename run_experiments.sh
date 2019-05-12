#!/usr/bin/env bash
env_name1=$1
env_name2=$2
env_name3=$3

for env_name in ${env_name1} ${env_name2} ${env_name3}; do
#    if [ -z env_name ]; then
#        echo "=> No more environemnt to run. Terminating..."
#        break
#    fi

    echo "=> Running environment ${env_name}"
    for random_seed in 1234 2314 2345 1235; do
        python main.py -c configs/algos/slbo.yml configs/envs/${env_name}.yml -s log_dir=experiments/${env_name}_${random_seed} seed=${random_seed}
    done

done