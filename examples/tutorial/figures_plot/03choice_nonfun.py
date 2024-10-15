import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def Individual(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    threshold = 0.6
    replacement_value = 0.605

    y1 = np.where(df['LABEL'].values < threshold, replacement_value, df['LABEL'].values)
    y2 = np.where(df['Top-K'].values < threshold, replacement_value, df['Top-K'].values)
    y3 = np.where(df['APS'].values < threshold, replacement_value, df['APS'].values)
    y4 = np.where(df['RAPS'].values < threshold, replacement_value, df['RAPS'].values)
    y5 = np.where(df['SYS'].values < threshold, replacement_value, df['SYS'].values)

    s1 = np.where(df['sLABEL'].values < threshold, replacement_value, df['sLABEL'].values)
    s2 = np.where(df['sTop-K'].values < threshold, replacement_value, df['sTop-K'].values)
    s3 = np.where(df['sAPS'].values < threshold, replacement_value, df['sAPS'].values)
    s4 = np.where(df['sRAPS'].values < threshold, replacement_value, df['sRAPS'].values)
    s5 = np.where(df['sSYS'].values < threshold, replacement_value, df['sSYS'].values)

    b1 = np.where(df['bLABEL'].values < threshold, replacement_value, df['bLABEL'].values)
    b2 = np.where(df['bTop-K'].values < threshold, replacement_value, df['bTop-K'].values)
    b3 = np.where(df['bAPS'].values < threshold, replacement_value, df['bAPS'].values)
    b4 = np.where(df['bRAPS'].values < threshold, replacement_value, df['bRAPS'].values)
    b5 = np.where(df['bSYS'].values < threshold, replacement_value, df['bSYS'].values)

    fig = plt.figure(figsize=(9, 5), dpi=80)
    ax = fig.add_axes([0.12, 0.25, 0.8, 0.5], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1) + 0.28)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 22,'family': 'Arial'})
    ax.set_yticks(np.arange(0.6, 1.001, 0.1))
    ax.set_yticklabels([0.6, 0.7, 0.8,0.9,1],
                       fontdict={'horizontalalignment': 'right', 'size': 22,'family': 'Arial'})
    ax.set_ylim((0.6, 1.01))
    ax.set_ylabel('Metric value', fontsize=22,fontdict={'family': 'Arial'})

    bar_width = 0.12
    ax.bar(x=np.arange(len(x)), height=y1, label='LAC', width=bar_width,
           linewidth=1, edgecolor='#8cc983', color='#a2d39b', zorder=10, alpha=0.9, hatch="||")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.04, height=y2, label='Top-K', width=bar_width,
           linewidth=1, edgecolor='#4b9441', color='#60b454', zorder=10,
           alpha=0.9, hatch="//")
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.08, height=y3, label='APS', width=bar_width,
           linewidth=1, edgecolor='#33652c', color='#4b9441', zorder=10,
           alpha=0.9, hatch="\\\\")
    ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.12, height=y4, label='RAPS', width=bar_width,
           linewidth=1, edgecolor='#1b3518', color='#397031', zorder=10, hatch="-")
    ax.bar(x=np.arange(len(x)) + 4 * bar_width + 0.16, height=y5, label='PROM', width=bar_width,
           linewidth=1, color='#274d22', zorder=10)

    # 最大最小值
    ax.hlines(y=s1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.hlines(y=s2, xmin=np.arange(len(x)) + bar_width + 0.04 - 0.03, xmax=np.arange(len(x)) + bar_width + 0.04 + 0.03,
              color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=s3, xmin=np.arange(len(x)) + 2 * bar_width + 0.08 - 0.03,
              xmax=np.arange(len(x)) + 2 * bar_width + 0.08 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=s4, xmin=np.arange(len(x)) + 3 * bar_width + 0.12 - 0.03,
              xmax=np.arange(len(x)) + 3 * bar_width + 0.12 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=s5, xmin=np.arange(len(x)) + 4 * bar_width + 0.16 - 0.03,
              xmax=np.arange(len(x)) + 4 * bar_width + 0.16 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)

    ax.hlines(y=b1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.hlines(y=b2, xmin=np.arange(len(x)) + bar_width + 0.04 - 0.03, xmax=np.arange(len(x)) + bar_width + 0.04 + 0.03,
              color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=b3, xmin=np.arange(len(x)) + 2 * bar_width + 0.08 - 0.03,
              xmax=np.arange(len(x)) + 2 * bar_width + 0.08 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=b4, xmin=np.arange(len(x)) + 3 * bar_width + 0.12 - 0.03,
              xmax=np.arange(len(x)) + 3 * bar_width + 0.12 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=b5, xmin=np.arange(len(x)) + 4 * bar_width + 0.16 - 0.03,
              xmax=np.arange(len(x)) + 4 * bar_width + 0.16 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)

    ax.vlines(x=np.arange(len(x)), ymin=s1, ymax=b1, color='k', alpha=0.6, linewidth=2, zorder=12)
    ax.vlines(x=np.arange(len(x)) + 1 * bar_width + 0.04, ymin=s2, ymax=b2, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.vlines(x=np.arange(len(x)) + 2 * bar_width + 0.08, ymin=s3, ymax=b3, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.vlines(x=np.arange(len(x)) + 3 * bar_width + 0.12, ymin=s4, ymax=b4, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.vlines(x=np.arange(len(x)) + 4 * bar_width + 0.16, ymin=s5, ymax=b5, color='k', alpha=0.6, linewidth=2,
              zorder=12)


    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.02, 1.12), ncol=5, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=2)
    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.savefig(str(name) + '.pdf')
    plt.show()

# vul
Individual(r'E:\model_drift\fugures_plot\data\indiv_vul.xlsx',r'E:\model_drift\fugures_plot\figure\individual_vul')
# thread
Individual(r'E:\model_drift\fugures_plot\data\indiv_thread.xlsx',r'E:\model_drift\fugures_plot\figure\individual_thread')
#loop
Individual(r'E:\model_drift\fugures_plot\data\indiv_loop.xlsx',r'E:\model_drift\fugures_plot\figure\individual_loop')
#device
Individual(r'E:\model_drift\fugures_plot\data\indiv_device.xlsx',r'E:\model_drift\fugures_plot\figure\individual_device')