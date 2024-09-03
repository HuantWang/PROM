import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def all_IL(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['Ori'].values
    y2 = df['Drift'].values
    y3 = df['IL'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(2, 3.51, 0.5))
    ax.set_yticklabels([2,2.5,3,3.5],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((1.9, 3.51))
    ax.set_ylabel('Accuracy', fontsize=20)
    bar_width = 0.15

    ax.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
           edgecolor='#D3D3D3', linewidth=1, color='#ede5fb', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
           edgecolor='#a79bbe',
           linewidth=1, color='#b7a5d8', zorder=10, alpha=0.9, )
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
           edgecolor='#261a4d',
           linewidth=1, color='#473192', zorder=10, alpha=0.9)

    font = {'family':'Arial',
            'weight': 'light',
            'size': 16
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.15, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

all_IL(r'E:\model_drift\fugures_plot\data\all_IL.xlsx',r'E:\model_drift\fugures_plot\figure\IL_all')