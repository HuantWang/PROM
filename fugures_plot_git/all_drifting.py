import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def Individual(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    threshold = 0.6
    replacement_value = 0.605

    y1 = np.where(df['C4'].values < threshold, replacement_value, df['C4'].values)
    y2 = np.where(df['C2'].values < threshold, replacement_value, df['C2'].values)
    y3 = np.where(df['C3'].values < threshold, replacement_value, df['C3'].values)
    y4 = np.where(df['C1'].values < threshold, replacement_value, df['C1'].values)

    s1 = np.where(df['sC4'].values < threshold, replacement_value, df['sC4'].values)
    s2 = np.where(df['sC2'].values < threshold, replacement_value, df['sC2'].values)
    s3 = np.where(df['sC3'].values < threshold, replacement_value, df['sC3'].values)
    s4 = np.where(df['sC1'].values < threshold, replacement_value, df['sC1'].values)

    b1 = np.where(df['bC4'].values < threshold, replacement_value, df['bC4'].values)
    b2 = np.where(df['bC2'].values < threshold, replacement_value, df['bC2'].values)
    b3 = np.where(df['bC3'].values < threshold, replacement_value, df['bC3'].values)
    b4 = np.where(df['bC1'].values < threshold, replacement_value, df['bC1'].values)

    fig = plt.figure(figsize=(9, 5), dpi=80)
    ax = fig.add_axes([0.12, 0.25, 0.8, 0.5], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1) + 0.28)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 22,'family': 'Arial'})
    ax.set_yticks(np.arange(0.7, 1.001, 0.1))
    ax.set_yticklabels([70, 80,90,100],
                       fontdict={'horizontalalignment': 'right', 'size': 22,'family': 'Arial'})
    ax.set_ylim((0.75, 1.01))
    # ax.set_ylabel('Accuracy', fontsize=22,fontdict={'family': 'Arial'})

    bar_width = 0.15
    ax.bar(x=np.arange(len(x)), height=y1, label='Heterogeneous Mapping', width=bar_width,
           linewidth=1, edgecolor='#D3D3D3', color='#ede4fb', zorder=10, alpha=0.9, hatch="")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.04, height=y2, label='Loop Vectorization', width=bar_width,
           linewidth=1, edgecolor='#a79bbe', color='#c5b4e0', zorder=10,
           alpha=0.9, hatch="")
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.08, height=y3, label='Vulnerability Detection', width=bar_width,
           linewidth=1, edgecolor='#655784', color='#9c86c6', zorder=10,
           alpha=0.9, hatch="")
    ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.12, height=y4, label='Thread Coarsening', width=bar_width,
           linewidth=1, color='#735bac', zorder=10)

    # 最大最小值
    ax.hlines(y=s1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.hlines(y=s2, xmin=np.arange(len(x)) + bar_width + 0.04 - 0.03, xmax=np.arange(len(x)) + bar_width + 0.04 + 0.03,
              color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=s3, xmin=np.arange(len(x)) + 2 * bar_width + 0.08 - 0.03,
              xmax=np.arange(len(x)) + 2 * bar_width + 0.08 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=s4, xmin=np.arange(len(x)) + 3 * bar_width + 0.12 - 0.03,
              xmax=np.arange(len(x)) + 3 * bar_width + 0.12 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)

    ax.hlines(y=b1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.hlines(y=b2, xmin=np.arange(len(x)) + bar_width + 0.04 - 0.03, xmax=np.arange(len(x)) + bar_width + 0.04 + 0.03,
              color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=b3, xmin=np.arange(len(x)) + 2 * bar_width + 0.08 - 0.03,
              xmax=np.arange(len(x)) + 2 * bar_width + 0.08 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=b4, xmin=np.arange(len(x)) + 3 * bar_width + 0.12 - 0.03,
              xmax=np.arange(len(x)) + 3 * bar_width + 0.12 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)

    ax.vlines(x=np.arange(len(x)), ymin=s1, ymax=b1, color='k', alpha=0.6, linewidth=2, zorder=12)
    ax.vlines(x=np.arange(len(x)) + 1 * bar_width + 0.04, ymin=s2, ymax=b2, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.vlines(x=np.arange(len(x)) + 2 * bar_width + 0.08, ymin=s3, ymax=b3, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.vlines(x=np.arange(len(x)) + 3 * bar_width + 0.12, ymin=s4, ymax=b4, color='k', alpha=0.6, linewidth=2,
              zorder=12)



    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.05, 1.25), ncol=2, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=2)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

# vul
Individual(r'E:\model_drift\fugures_plot\data\drift_all.xlsx',r'E:\model_drift\fugures_plot\figure\all_drifting')