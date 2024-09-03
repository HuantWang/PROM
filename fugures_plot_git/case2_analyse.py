import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def drifting_loop_all(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值

    y4 = df['SVM'].values
    y5 = df['MLP'].values
    y6 = df['LSTM'].values
    y7 = df['GNN'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.12, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.55)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.8, 0.91, 0.05))
    ax.set_yticklabels([0.8,0.85, 0.9],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.8, 0.9))
    # ax.set_ylabel('% of value', fontsize=14)
    bar_width = 0.15


    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y4, label='SVM', width=bar_width,
           edgecolor='#b08cee',
           linewidth=1, color='#cfb8f5', zorder=10, alpha=0.9,hatch='')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y5, label='MLP', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9,hatch='////')
    ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.18, height=y6, label='LSTM', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9,hatch='|||')
    ax.bar(x=np.arange(len(x)) + 4 * bar_width + 0.24, height=y7, label='GNN', width=bar_width,
           edgecolor='#6a23e0',
           linewidth=1, color='#8850e6', zorder=10, alpha=0.9,hatch='-')



    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.03, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def IL_loop_all(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    # y1 = df['Ori'].values
    y2 = df['Drift'].values
    y3 = df['IL'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.2)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(1, 1.61, 0.2))
    ax.set_yticklabels([1,1.2,1.4,1.6],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((1, 1.7))
    ax.set_ylabel('Speedup', fontsize=20)
    bar_width = 0.15

    # ax.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
    #        edgecolor='#D3D3D3', linewidth=1, color='#ede5fb', zorder=10,
    #        alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
           edgecolor='#a79bbe',
           linewidth=1, color='#c5b4e0', zorder=10, alpha=0.9, )
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
           edgecolor='#261a4d',
           linewidth=1, color='#735bac', zorder=10, alpha=0.9)

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.15, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()


# drifting_loop_all(r'E:\model_drift\fugures_plot\data\drifting_loop_all.xlsx',
#                   r'E:\model_drift\fugures_plot\figure\drifting_loop_all')

# IL_loop_all(r'E:\model_drift\fugures_plot\data\IL_loop_all.xlsx',r'E:\model_drift\fugures_plot\figure\IL_loop_all')