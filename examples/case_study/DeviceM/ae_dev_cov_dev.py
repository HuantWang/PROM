import sys
import warnings
warnings.filterwarnings("ignore")


sys.path.append('../case_study/DeviceM/')
from ae_DevM_i2v_cd import ae_dev_i2v
from ae_DevM_Deeptune_cd import ae_dev_deep
from ae_DevM_Programl_cd import ae_dev_programl


print("\nThe evaluation on Instruct2vec\n")
ae_dev_i2v()

print("\nThe evaluation on DeepTune\n")
ae_dev_deep(eva_flag="cd")

print("\nThe evaluation on PrograML\n")
ae_dev_programl(eva_flag="cd")