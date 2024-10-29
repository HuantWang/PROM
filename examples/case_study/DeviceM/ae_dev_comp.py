
import sys

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