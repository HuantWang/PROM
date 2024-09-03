import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

def drifting_thread(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['Ori'].values
    y2 = df['Drift'].values
    # y3 = df['IL'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.18, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 3, 1)+0.1)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(0.85, 1.26, 0.1))
    ax.set_yticklabels([0.85,0.95,1.05,1.15,1.25],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((0.8, 1.25))
    ax.set_ylabel('Speedup', fontsize=20)
    bar_width = 0.15

    ax.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
           edgecolor='#a79bbe', linewidth=1, color='#b7a5d8', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
           edgecolor='#261a4d',
           linewidth=1, color='#473192', zorder=10, alpha=0.9, )
    # ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #        edgecolor='#261a4d',
    #        linewidth=1, color='#473192', zorder=10, alpha=0.9)

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0.1, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def drifting_loop(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['Ori'].values
    y2 = df['Drift'].values
    # y3 = df['IL'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.18, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 3, 1)+0.1)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(1.2, 1.61, 0.1))
    ax.set_yticklabels([1.2,1.3,1.4,1.5,1.6],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((1.2, 1.61))
    ax.set_ylabel('Speedup', fontsize=20)
    bar_width = 0.15

    ax.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
           edgecolor='#a79bbe', linewidth=1, color='#b7a5d8', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
           edgecolor='#261a4d',
           linewidth=1, color='#473192', zorder=10, alpha=0.9, )
    # ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #        edgecolor='#261a4d',
    #        linewidth=1, color='#473192', zorder=10, alpha=0.9)

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0.1, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def drifting_dev(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # 平均值
    y1 = df['Ori'].values
    y2 = df['Drift'].values
    # y3 = df['IL'].values

    fig = plt.figure(figsize=(6, 3), dpi=80)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.6], zorder=11)

    ax.set_xticks(np.arange(0, 4, 1)+0.1)
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    ax.set_yticks(np.arange(2, 3.51, 0.5))
    ax.set_yticklabels([2,2.5,3,3.5],
                       fontdict={'horizontalalignment': 'right', 'size': 20,'family': 'Arial'})
    ax.set_ylim((1.9, 3.51))
    ax.set_ylabel('Speedup', fontsize=20)
    bar_width = 0.15

    ax.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
           edgecolor='#a79bbe', linewidth=1, color='#b7a5d8', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
           edgecolor='#261a4d',
           linewidth=1, color='#473192', zorder=10, alpha=0.9, )
    # ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #        edgecolor='#261a4d',
    #        linewidth=1, color='#473192', zorder=10, alpha=0.9)

    font = {'family':'Arial',
            'weight': 'light',
            'size': 20
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0.1, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()


def drifting_vul(path, name):
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
           edgecolor='#a79bbe', linewidth=1, color='#b7a5d8', zorder=10,
           alpha=0.9, align="center")
    ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
           edgecolor='#261a4d',
           linewidth=1, color='#473192', zorder=10, alpha=0.9, )
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


def drifting_vul_modified(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    y1 = df['Ori'].values
    y2 = df['Drift'].values
    # y3 = df['IL'].values


    # Set up the figure and axes with custom height ratios
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 3), dpi=80,
                                   gridspec_kw={'height_ratios': [1.5, 1]})
    fig.subplots_adjust(hspace=0.001)  # adjust the space between the axes

    bar_width = 0.15

    # Plot the same data on both axes
    ax1.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
            edgecolor='#a79bbe', linewidth=1, color='#b7a5d8', alpha=0.9, align="center")
    ax1.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
            edgecolor='#261a4d', linewidth=1, color='#473192', alpha=0.9)
    # ax1.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #         edgecolor='#261a4d', linewidth=1, color='#473192', alpha=0.9)

    ax2.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
            edgecolor='#a79bbe', linewidth=1, color='#b7a5d8', alpha=0.9, align="center")
    ax2.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
            edgecolor='#261a4d', linewidth=1, color='#473192', alpha=0.9)
    # ax2.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #         edgecolor='#261a4d', linewidth=1, color='#473192', alpha=0.9)

    # Set the y limits
    ax1.set_ylim(0.8, 1.01)
    ax2.set_ylim(0.1, 0.5)

    # Remove spines and ticks of the top and bottom subplot, respectively
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Apply broken axis lines
    d = .015  # size of the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Set xticks and yticks
    ax1.set_yticks(np.arange(0.8, 1.01, 0.1))
    ax1.set_yticklabels([0.8, 0.9, 1.0],
                        fontdict={'horizontalalignment': 'right', 'size': 20, 'family': 'Arial'})
    ax2.set_yticks(np.arange(0.1, 0.51, 0.1))

    ax2.set_xticks(np.arange(0, 4, 1) + 0.05)
    ax2.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20, 'family': 'Arial'})

    ax2.set_yticks(np.arange(0, 0.51, 0.2))
    ax2.set_yticklabels([0,0.2, 0.4],
                        fontdict={'horizontalalignment': 'right', 'size': 20, 'family': 'Arial'})

    # ax1.set_ylabel('Accuracy', fontsize=20)
    y_label = ax1.set_ylabel('Accuracy', fontsize=20)
    y_label.set_position((y_label.get_position()[0], y_label.get_position()[1] - 0.35))
    font = {'family': 'Arial', 'weight': 'light', 'size': 20}

    # Legend
    box = ax1.get_position()
    ax1.set_position([box.x0+0.05, box.y0+0.02, box.width, box.height * 0.8])
    box = ax2.get_position()
    ax2.set_position([box.x0 + 0.05, box.y0+0.02, box.width, box.height * 0.8])
    ax1.legend(loc='center left', bbox_to_anchor=(0.1, 1.22), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
               handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.savefig(str(name) + '.pdf')
    plt.show()


drifting_dev(r'E:\model_drift\fugures_plot\data\IL_dev.xlsx',r'E:\model_drift\fugures_plot\figure\drifting_dev')
# drifting_loop(r'E:\model_drift\fugures_plot\data\IL_loop.xlsx',r'E:\model_drift\fugures_plot\figure\drifting_loop')
# drifting_thread(r'E:\model_drift\fugures_plot\data\IL_thread.xlsx',r'E:\model_drift\fugures_plot\figure\drifting_thread')
# drifting_vul_modified(r'E:\model_drift\fugures_plot\data\IL_vul.xlsx',r'E:\model_drift\fugures_plot\figure\drifting_vul')