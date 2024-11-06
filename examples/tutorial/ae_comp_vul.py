
import sys



sys.path.append('../case_study/BugD/')
from ae_VD_codebert_com import ae_vul_codebert
from ae_VD_linevul_com import ae_vul_linevul
from ae_VD_vulde_com import ae_vul_vulde

print("\nCase 4:\n")
print("\n--- Comparison Evaluation: CodeBERT ---\n")
ae_vul_codebert()

print("\n--- Comparison Evaluation: Linevul ---\n")
ae_vul_linevul()
#
print("\n--- Comparison Evaluation: VUlDE ---\n")
ae_vul_vulde()
