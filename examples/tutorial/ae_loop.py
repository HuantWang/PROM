import subprocess

# setting the environments
subprocess.run(["bash", "env.sh", "C1"])

import sys
sys.path.append('../case_study/Loop/')
from ae_loop_SVM import ae_loop_svm_script
from ae_loop_ma import ae_loop_ma_script
from ae_loop_de import ae_loop_de_script
# from ae_thread_i2v import ae_thread_i2v_script


ae_loop_svm_script()
ae_loop_ma_script()
ae_loop_de_script()

# from figures_plot.ae_loop_plot import ae_plot
#
# ae_plot()