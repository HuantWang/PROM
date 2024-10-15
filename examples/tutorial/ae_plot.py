import sys
sys.path.append('../case_study/Thread/')
from figures_plot.ae_thread_plot import ae_plot
import logging
logging.disable(logging.CRITICAL)

ae_plot('thread')