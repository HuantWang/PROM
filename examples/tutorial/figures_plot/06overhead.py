import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def overhead(path, name):
    df = pd.read_excel(path)
    x = df['model'].values
    # 平均值
    y1 = df['Train'].values
    y2 = df['Retrain'].values


    fig = plt.figure(figsize=(10, 4), dpi=80)
    ax = fig.add_axes([0.2, 0.3, 0.7, 0.5], zorder=11)

    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_yticklabels(x,fontdict={'size': 22, 'family': 'Arial'})
    ax.set_xticks(np.arange(0.0, 32.01, 6))
    ax.set_xticklabels([0, 6, 12, 18,24,30],
                       fontdict={'horizontalalignment': 'center', 'size': 22, 'family': 'Arial'})
    ax.set_xlim((0.0, 32.01))
    ax.set_xlabel('Overhead (hours)', fontsize=22)
    ax.set_ylabel('Case study',fontsize=22)
    height = 0.6
    ax.barh(y=np.arange(len(x)), width=y1, height=height, label='Initial Training',
            edgecolor='#4b9441',
            linewidth=1, color='#a2d39b', alpha=1, align="center")
    ax.barh(y=np.arange(len(x)), width=y2, height=height, left=y1, label='Incremental Learning',
            edgecolor='#33652c',
            linewidth=1, color='#4b9441', zorder=10, alpha=0.9)



    font = {'family': 'Arial',
            'weight': 'light',
            'size': 20
            }
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    ax.legend(loc='center left', bbox_to_anchor=(-0.05, 1.15), ncol=2, prop=font, columnspacing=2,
              handletextpad=0.6,
              handlelength=3, )

    plt.savefig(str(name) + '.pdf')

    plt.show()

path = r"E:\model_drift\fugures_plot\data\overhead.xlsx"
overhead(path, r'E:\model_drift\fugures_plot\figure\overhead')