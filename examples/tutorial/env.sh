#!/bin/bash


source ~/anaconda3/etc/profile.d/conda.sh  # 请确保路径正确


if [ -z "$1" ]; then
    echo "No environment specified. Please provide either 'C1' or 'C5' as the environment name."
    echo "Usage: ./run_conda_env.sh <C1|C5>"
    exit 1
fi

if [ "$1" == "C1" ]; then
    conda activate thread
    echo "Environment for 'C1,C2,C4' activated."
elif [ "$1" == "C3" ]; then
    conda activate dev
    echo "Environment for 'C3' activated."
elif [ "$1" == "C5" ]; then
    conda activate tvm
    echo "Environment for 'C5' activated."
fi
