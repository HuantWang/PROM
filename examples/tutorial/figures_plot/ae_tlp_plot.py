import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import logging
logging.disable(logging.CRITICAL)





def drifting_tensor(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['BERT_large'].values
    y2 = df['BERT_medium'].values
    y3 = df['BERT_tiny'].values

    fig = plt.figure(figsize=(8, 3), dpi=80)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.8, 1.01, 0.05))
    ax.set_yticklabels([0.8,0.85, 0.9,0.95, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.8, 1.01))
    ax.set_ylabel('Metric value', fontsize=20)
    bar_width = 0.2

    ax.bar(x=np.arange(len(x)), height=y1, label='BERT_large', width=bar_width,
           edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='BERT_medium', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9, hatch='////')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='BERT_tiny', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0, 1.14), ncol=6, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

from tabulate import tabulate

def draw_table():
    data = [
        ["Native deployment", 0.845, 0.224, 0.668, 0.642],
        ["PROM assisted deployment", "/", 0.794, 0.871, 0.843]
    ]
    headers = ["Network", "BERT-base", "BERT-tiny", "BERT-medium", "BERT-large"]

    table = tabulate(data, headers=headers, tablefmt="grid", stralign="center", numalign="center")
    print(table)


def ae_tlp_plot_script(case=''):
    drifting_tensor(r'./figures_plot/data/drifting_tensor.xlsx',
                    r'./figures_plot/figure/detectdrifting_tensor')

    print("Figure 8(e) C5: DNN code generation. Prom’s performance for detecting drifting samples across case studies and underlying models.")

    draw_table()

    print(" Table 3. C5: DNN code generation (performance to the oracle ratio) - trained on BERT-base and tested on BERT variants.")

# ae_vul_plot_script()
# violin_3(method='thread')