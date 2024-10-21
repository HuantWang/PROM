import sys
import argparse
import logging

# Add the path to the case study modules
sys.path.append('../case_study/Thread/')
from figures_plot.ae_thread_plot import ae_thread_plot_script
from figures_plot.ae_loop_plot import ae_loop_plot_script
from figures_plot.ae_dev_plot import ae_dev_plot_script
from figures_plot.ae_vul_plot import ae_vul_plot_script
from figures_plot.ae_tlp_plot import ae_tlp_plot_script
# Disable logging below CRITICAL level
logging.disable(logging.CRITICAL)

def plot_script(case=''):
    if case == 'thread':
        ae_thread_plot_script()
    elif case == 'loop':
        ae_loop_plot_script()
    elif case == 'dev':
        ae_dev_plot_script()
    elif case == 'vul':
        ae_vul_plot_script()
    elif case == 'tlp':
        ae_tlp_plot_script()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run plot script for specific case.")
    parser.add_argument(
        '--case',
        type=str,
        choices=['thread', 'loop','dev','vul','tlp'],
        help="Specify the case to run"
    )
    args = parser.parse_args()

    # Call the plot script with the specified case
    # plot_script(case=args.case)
    # plot_script('tlp')
