import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

import matplotlib
def drifting_thread(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['Magni'].values
    y2 = df['Deeptune'].values
    y3 = df['IR2Vec'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.17, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.8, 1.01, 0.05))
    ax.set_yticklabels([0.8,0.85, 0.9,0.95, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.8, 1.01))
    ax.set_ylabel('Metric value', fontsize=20)
    bar_width = 0.2

    ax.bar(x=np.arange(len(x)), height=y1, label='Magni', width=bar_width,
           edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='DeepTune', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9, hatch='////')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='IR2Vec', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0, 1.2), ncol=6, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def drifting_loop(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['K'].values
    y2 = df['Magni'].values
    y3 = df['Deeptune'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.7, 1.01, 0.1))
    ax.set_yticklabels([0.7,0.8,0.9, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.7, 1.01))
    ax.set_ylabel('Metric value', fontsize=20)
    bar_width = 0.2

    ax.bar(x=np.arange(len(x)), height=y1, label='K.Stock et al.', width=bar_width,
           edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Magni', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9, hatch='////' )
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='DeepTune', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.13, 1.2), ncol=6, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def drifting_vul(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['codebert'].values
    y2 = df['linevul'].values
    y3 = df['vuldee'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.17, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.85, 1.01, 0.05))
    ax.set_yticklabels([0.85, 0.9,0.95, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.85, 1.01))
    ax.set_ylabel('Metric value', fontsize=20)
    bar_width = 0.2

    ax.bar(x=np.arange(len(x)), height=y1, label='CodeXGLUE', width=bar_width,
           edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Linevul', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9,  hatch='////')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='Vulde', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')
    # ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.18, height=y4, label='BERT_base', width=bar_width,
    #        edgecolor='#6a23e0',
    #        linewidth=1, color='#8850e6', zorder=10, alpha=0.9, hatch='-')

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.02, 1.17), ncol=3, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()


def drifting_dev(path, name):

    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['ir2v'].values
    y2 = df['programl'].values
    y3 = df['deeptune'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.6], zorder=11)



    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.5, 1.01, 0.2))
    ax.set_yticklabels([0.5, 0.7,0.9],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.5, 1))
    ax.set_ylabel('Metric value', fontsize=20)
    bar_width = 0.15

    ax.bar(x=np.arange(len(x)), height=y1, label='IR2Vec', width=bar_width,
           edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Programl', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9,  hatch='////')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='Deeptune', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')


    font = {'family':'Arial',
            'weight': 'light',
            'size': 18
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.0, 1.15), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def drifting_tensor(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['BERT_large'].values
    y2 = df['BERT_medium'].values
    y3 = df['BERT_tiny'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.12, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.3)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.8, 1.01, 0.05))
    ax.set_yticklabels([0.8,0.85, 0.9,0.95, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.8, 1.01))
    # ax.set_ylabel('% of value', fontsize=14)
    bar_width = 0.15

    ax.bar(x=np.arange(len(x)), height=y1, label='LineVul', width=bar_width,
           edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Vulde', width=bar_width,
           edgecolor='#925fe8',
           linewidth=1, color='#b08cee', zorder=10, alpha=0.9,  hatch='////')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='Codexglue', width=bar_width,
           edgecolor='#7e41e4',
           linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')
    ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.18, height=y4, label='FUNDED', width=bar_width,
           edgecolor='#6a23e0',
           linewidth=1, color='#8850e6', zorder=10, alpha=0.9, hatch='-')

    font = {'family':'Arial',
            'weight': 'light',
            'size': 18
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(-0.18, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def drifting_pos(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['C1'].values
    y2 = df['C2'].values
    y3 = df['C3'].values
    y4 = df['C4'].values
    y5 = df['C5'].values  # 添加对C5列的处理

    fig = plt.figure(figsize=(8, 3), dpi=80)
    ax = fig.add_axes([0.12, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.45)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.85, 1.01, 0.05))
    ax.set_yticklabels([0.85, 0.9,0.95, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.85, 1.01))
    # ax.set_ylabel('% of value', fontsize=14)
    bar_width = 0.12

    ax.bar(x=np.arange(len(x)), height=y1, label='C1', width=bar_width,
           edgecolor='#1a2a0e', linewidth=1, color='#bcd1ac', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='C2', width=bar_width,
           edgecolor='#1a2a0e',
           linewidth=1, color='#8aaf6e', zorder=10, alpha=0.9,  hatch='////')
    ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='C3', width=bar_width,
           edgecolor='#1a2a0e',
           linewidth=1, color='#588e31', zorder=10, alpha=0.9, hatch='|||')
    ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.18, height=y4, label='C4', width=bar_width,
           edgecolor='#1a2a0e',
           linewidth=1, color='#3d6322', zorder=10, alpha=0.9, hatch='-')
    ax.bar(x=np.arange(len(x)) + 4 * bar_width + 0.24, height=y5, label='C5', width=bar_width,
           edgecolor='#1a2a0e',
           linewidth=1, color='#2c4718', zorder=10, alpha=0.9, hatch='\\')  # 添加新的bar

    font = {'family':'Arial',
            'weight': 'light',
            'size': 18
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0.15, 1.2), ncol=5, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

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

# drifting_tensor(r'E:\model_drift\fugures_plot\data\drifting_tensor.xlsx',
#                 r'E:\model_drift\fugures_plot\figure\detectdrifting_tensor')
#thread
# drifting_thread(r'E:\model_drift\fugures_plot\data\drifting_thread.xlsx',r'E:\model_drift\fugures_plot\figure\detectdrifting_thread')

#loop
# drifting_loop(r'E:\model_drift\fugures_plot\data\drifting_loop.xlsx',r'E:\model_drift\fugures_plot\figure\detectdrifting_loop')
#vul
# drifting_vul(r'E:\model_drift\fugures_plot\data\drifting_vul.xlsx',r'E:\model_drift\fugures_plot\figure\detectdrifting_vul')
#dev
# drifting_dev(r'E:\model_drift\fugures_plot\data\drifting_dev.xlsx',r'E:\model_drift\fugures_plot\figure\detectdrifting_dev')

# drifting_pos(r'E:\model_drift\fugures_plot\data\drifting_poster.xlsx',r'E:\model_drift\fugures_plot\figure\drifting_poster')