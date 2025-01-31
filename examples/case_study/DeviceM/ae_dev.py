import sys
sys.path.append('../case_study/DeviceM/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ae_DevM_i2v import ae_dev_i2v
from ae_DevM_Deeptune import ae_dev_deep
from ae_DevM_Programl import ae_dev_programl
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from absl import logging
logging.set_verbosity(logging.ERROR)


print("\nThe evaluation on Instruct2vec\n")
ae_dev_i2v()

print("\nThe evaluation on DeepTune\n")
ae_dev_deep()

print("\nThe evaluation on PrograML\n")
ae_dev_programl()
#
# import sys
# sys.path.append('../case_study/Thread/')
# from figures_plot.ae_thread_plot import ae_plot
#
# ae_plot('thread')