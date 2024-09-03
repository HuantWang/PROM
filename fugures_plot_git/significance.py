import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def Individual(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    threshold = 0.5
    replacement_value = 0.505

    y2 = np.where(df['RAISE'].values < threshold, replacement_value, df['RAISE'].values)
    y1 = np.where(df['TESSERACT'].values < threshold, replacement_value, df['TESSERACT'].values)
    y3 = np.where(df['PROM'].values < threshold, replacement_value, df['PROM'].values)

    s2 = np.where(df['sRAISE'].values < threshold, replacement_value, df['sRAISE'].values)
    s1 = np.where(df['sTESSERACT'].values < threshold, replacement_value, df['sTESSERACT'].values)
    s3 = np.where(df['sPROM'].values < threshold, replacement_value, df['sPROM'].values)

    b2 = np.where(df['bRAISE'].values < threshold, replacement_value, df['bRAISE'].values)
    b1 = np.where(df['bTESSERACT'].values < threshold, replacement_value, df['bTESSERACT'].values)
    b3 = np.where(df['bPROM'].values < threshold, replacement_value, df['bPROM'].values)

    fig = plt.figure(figsize=(8, 4), dpi=80)
    ax = fig.add_axes([0.12, 0.25, 0.8, 0.5], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1) + 0.2)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 22,'family': 'Arial'})
    ax.set_yticks(np.arange(0.6, 1.001, 0.1))
    ax.set_yticklabels([0.6, 0.7, 0.8,0.9,1],
                       fontdict={'horizontalalignment': 'right', 'size': 22,'family': 'Arial'})
    ax.set_ylim((0.6, 1.01))
    # ax.set_ylabel('Accuracy', fontsize=22,fontdict={'family': 'Arial'})

    bar_width = 0.15
    ax.bar(x=np.arange(len(x)), height=y1, label='TESSERACT', width=bar_width,
           linewidth=1, edgecolor='#8cc983', color='#a2d39b', zorder=10, alpha=0.9, hatch="||")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.04, height=y2, label='RISE', width=bar_width,
           linewidth=1, edgecolor='#4b9441', color='#60b454', zorder=10,
           alpha=0.9, hatch="//")
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.08, height=y3, label='PROM', width=bar_width,
           linewidth=1, edgecolor='#33652c', color='#4b9441', zorder=10,
           alpha=0.9, hatch="\\\\")

    # 最大最小值
    ax.hlines(y=s1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.hlines(y=s2, xmin=np.arange(len(x)) + bar_width + 0.04 - 0.03, xmax=np.arange(len(x)) + bar_width + 0.04 + 0.03,
              color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=s3, xmin=np.arange(len(x)) + 2 * bar_width + 0.08 - 0.03,
              xmax=np.arange(len(x)) + 2 * bar_width + 0.08 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)

    ax.hlines(y=b1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.hlines(y=b2, xmin=np.arange(len(x)) + bar_width + 0.04 - 0.03, xmax=np.arange(len(x)) + bar_width + 0.04 + 0.03,
              color='k', alpha=0.6, linewidth=2.5, zorder=12)
    ax.hlines(y=b3, xmin=np.arange(len(x)) + 2 * bar_width + 0.08 - 0.03,
              xmax=np.arange(len(x)) + 2 * bar_width + 0.08 + 0.03, color='k', alpha=0.6, linewidth=2.5, zorder=12)

    ax.vlines(x=np.arange(len(x)), ymin=s1, ymax=b1, color='k', alpha=0.6, linewidth=2, zorder=12)
    ax.vlines(x=np.arange(len(x)) + 1 * bar_width + 0.04, ymin=s2, ymax=b2, color='k', alpha=0.6, linewidth=2,
              zorder=12)
    ax.vlines(x=np.arange(len(x)) + 2 * bar_width + 0.08, ymin=s3, ymax=b3, color='k', alpha=0.6, linewidth=2,
              zorder=12)



    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    y_label = ax.set_ylabel('F1 score', fontsize=20)
    y_label.set_position((y_label.get_position()[0], y_label.get_position()[1]))
    font = {'family': 'Arial', 'weight': 'light', 'size': 20}
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0., 1.15), ncol=3, prop=font, columnspacing=1, handletextpad=0.5,
              handlelength=2)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

# vul
Individual(r'E:\model_drift\fugures_plot\data\significance_compare.xlsx',r'E:\model_drift\fugures_plot\figure\sig')