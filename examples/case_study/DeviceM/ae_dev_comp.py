import subprocess

# setting the environments
subprocess.run(["bash", "env.sh", "C1"])
import sys
# sys.path.append('../case_study/Thread/')

# from ae_thread_deep_com import ae_thread_deep_script
# from ae_thread_magni_com import ae_thread_magni_script
# from ae_thread_i2v_com import ae_thread_i2v_script


#
#
# print("\nCase 1:\n")
# print("\n--- Comparison Evaluation: DeepTune ---\n")
# ae_thread_deep_script()
#
# print("\n--- Comparison Evaluation: Magni ---\n")
# ae_thread_magni_script()
#
# print("\n--- Comparison Evaluation: Instruction2Vec ---\n")
# ae_thread_i2v_script()


# sys.path.append('../case_study/Loop/')
# from ae_loop_SVM_com import ae_loop_svm_script
# from ae_loop_ma_com import ae_loop_ma_script
# from ae_loop_de_com import ae_loop_de_script
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


# sys.path.append('../case_study/BugD/')
# from ae_VD_codebert_com import ae_vul_codebert
# from ae_VD_linevul_com import ae_vul_linevul
# from ae_VD_vulde_com import ae_vul_vulde
#
# print("\nCase 2:\n")
# print("\n--- Comparison Evaluation: CodeBERT ---\n")
# ae_vul_codebert()
#
# print("\n--- Comparison Evaluation: Linevul ---\n")
# ae_vul_linevul()
#
# print("\n--- Comparison Evaluation: VUlDE ---\n")
# ae_vul_vulde()


sys.path.append('../case_study/DeviceM/')
from ae_DevM_i2v_com import ae_dev_i2v
from ae_DevM_Deeptune_com import ae_dev_deep
from ae_DevM_Programl_com import ae_dev_programl


print("\nThe evaluation on Instruct2vec\n")
ae_dev_i2v()

print("\nThe evaluation on DeepTune\n")
ae_dev_deep(eva_flag="comapre")

print("\nThe evaluation on PrograML\n")
ae_dev_programl(eva_flag="comapre")