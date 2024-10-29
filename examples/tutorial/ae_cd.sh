#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

python ae_cd_thread.py
python ae_cd_loop.py
python ae_cd_vul.py

conda activate dev
echo "Environment for 'C3' activated."
cd ../case_study/DeviceM/
python ae_dev_cov_dev.py
cd ../../tutorial/

conda activate tvm
python ae_cd_vul.py