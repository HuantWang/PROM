#!/bin/bash


source ~/anaconda3/etc/profile.d/conda.sh

conda activate tvm
echo "Environment for 'C5' activated."
python ae_tlp.py
