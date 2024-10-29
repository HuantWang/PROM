#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

python ae_comp_thread.py
python ae_comp_loop.py
python ae_comp_vul.py

conda activate dev
echo "Environment for 'C3' activated."
cd ../case_study/DeviceM/
python ae_dev_comp.py
cd ../../tutorial/
