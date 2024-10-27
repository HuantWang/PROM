import subprocess

# setting the environments
subprocess.run(["bash", "env.sh", "C1"])
import sys
sys.path.append('../case_study/Thread/')
from ae_thread_deep import ae_thread_deep_script
from ae_thread_magni import ae_thread_magni_script
from ae_thread_i2v import ae_thread_i2v_script
#
#

print("\nThe evaluation on DeepTune\n")
ae_thread_deep_script()

print("\nThe evaluation on Magni\n")
ae_thread_magni_script()

print("\nThe evaluation on Instruction2Vec\n")
ae_thread_i2v_script()


#
# import sys
# sys.path.append('../case_study/Thread/')
# from figures_plot.ae_thread_plot import ae_plot
#
# ae_plot('thread')