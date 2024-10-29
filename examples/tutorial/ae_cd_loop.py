import sys
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
