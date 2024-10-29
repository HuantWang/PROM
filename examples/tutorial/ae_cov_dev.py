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



# sys.path.append('../case_study/Loop/')
# from ae_loop_SVM_cd import ae_loop_svm_script
# from ae_loop_ma_cd import ae_loop_ma_script
# from ae_loop_de_cd import ae_loop_de_script
#
# print("\nCase 2:\n")
# print("\n--- Comparison Evaluation: DeepTune ---\n")
# ae_loop_de_script()
#
# print("\n--- Comparison Evaluation: Magni ---\n")
# ae_loop_ma_script()
#
# print("\n--- Comparison Evaluation: K.Stock ---\n")
# ae_loop_svm_script()

# import sys
# sys.path.append('../case_study/BugD/')
#
# from ae_VD_codebert_cd import ae_vul_codebert
# from ae_VD_linevul_cd import ae_vul_linevul
# from ae_VD_vulde_cd import ae_vul_vulde
#
# print("\nCase 4:\n")
# print("\n--- Comparison Evaluation: CodeBERT ---\n")
# ae_vul_codebert()
#
# print("\n--- Comparison Evaluation: Linevul ---\n")
# ae_vul_linevul()
#
# print("\n--- Comparison Evaluation: VUlDE ---\n")
# ae_vul_vulde()


sys.path.append('../case_study/DeviceM/')
from ae_DevM_i2v_cd import ae_dev_i2v
# from ae_DevM_Deeptune import ae_dev_deep
# from ae_DevM_Programl import ae_dev_programl


print("\nThe evaluation on Instruct2vec\n")
ae_dev_i2v()

# print("\nThe evaluation on DeepTune\n")
# ae_dev_deep()

# print("\nThe evaluation on PrograML\n")
# ae_dev_programl()