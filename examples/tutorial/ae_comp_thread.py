
import sys
sys.path.append('../case_study/Thread/')

from ae_thread_deep_com import ae_thread_deep_script
from ae_thread_magni_com import ae_thread_magni_script
from ae_thread_i2v_com import ae_thread_i2v_script

print("\nCase 1:\n")
print("\n--- Comparison Evaluation: DeepTune ---\n")
ae_thread_deep_script()

print("\n--- Comparison Evaluation: Magni ---\n")
ae_thread_magni_script()

print("\n--- Comparison Evaluation: Instruction2Vec ---\n")
ae_thread_i2v_script()
