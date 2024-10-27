import subprocess

# setting the environments
subprocess.run(["bash", "env.sh", "C1"])
import sys
# sys.path.append('../case_study/Thread/')
# from ae_thread_deep_cd import ae_thread_deep_script
# from ae_thread_magni_cd import ae_thread_magni_script
# from ae_thread_i2v_cd import ae_thread_i2v_script
#
#

# print("\nThe coverage deviation evaluation on DeepTune\n")
# ae_thread_deep_script()
#
# print("\nThe coverage deviation evaluation on Magni\n")
# ae_thread_magni_script()
#
# print("\nThe coverage deviation evaluation on Instruction2Vec\n")
# ae_thread_i2v_script()



sys.path.append('../case_study/Loop/')
from ae_loop_SVM_cd import ae_loop_svm_script
from ae_loop_ma_cd import ae_loop_ma_script
from ae_loop_de_cd import ae_loop_de_script

print("\nCase 2:\n")
print("\n--- Comparison Evaluation: DeepTune ---\n")
ae_loop_de_script()

print("\n--- Comparison Evaluation: Magni ---\n")
ae_loop_ma_script()

print("\n--- Comparison Evaluation: K.Stock ---\n")
ae_loop_svm_script()