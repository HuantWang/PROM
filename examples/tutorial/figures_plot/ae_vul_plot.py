import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import logging
logging.disable(logging.CRITICAL)


def set_box_colors(bp, colors, edge_colors):
    for i, (patch, color, edge_c) in enumerate(zip(bp['boxes'], colors, edge_colors)):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge_c)
        if i%2==1:
            patch.set_hatch('/////')

        # 设置当前箱体的中位数、whiskers、fliers和caps的颜色
        plt.setp(bp['medians'], color=edge_c, linewidth=0.1)
        plt.setp(bp['whiskers'], color=edge_c, linewidth=0.1)
        plt.setp(bp['fliers'], color=edge_c, linewidth=0.1)
        plt.setp(bp['caps'], color=edge_c, linewidth=0.1)
        for flier in bp['fliers']:
            # flier.set_markerfacecolor(edge_c)
            flier.set_markeredgecolor(edge_c)

# def set_line_colors(bp, color):
#     for patch, color in zip(bp['boxes'], color):


def set_violin_colors(vp, colors, edge_color, linewidth=0.7):
    for i, (pc, color, edge_c) in enumerate(zip(vp['bodies'], colors, edge_color)):
        pc.set_facecolor(color)
        pc.set_edgecolor(edge_c)
        pc.set_alpha(0.6)
        pc.set_linewidths(linewidth)

        plt.setp(vp['cmedians'], color=edge_c, linewidth=linewidth)
        plt.setp(vp['cmaxes'], color=edge_c, linewidth=linewidth)
        plt.setp(vp['cmins'], color=edge_c, linewidth=linewidth)
        plt.setp(vp['cbars'], color=edge_c, linewidth=linewidth)
        # 设置异常点颜色




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
    ax.legend(loc='center left', bbox_to_anchor=(-0.05, 1.17), ncol=3, prop=font, columnspacing=0.5, handletextpad=0.1,
              handlelength=1.15)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    plt.show()

def Individual(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    threshold = 0.6
    minimum_value = 0.605

    y1 = np.where(df['LABEL'].values < threshold, minimum_value, df['LABEL'].values)
    y2 = np.where(df['Top-K'].values < threshold, minimum_value, df['Top-K'].values)
    y3 = np.where(df['APS'].values < threshold, minimum_value, df['APS'].values)
    y4 = np.where(df['RAPS'].values < threshold, minimum_value, df['RAPS'].values)
    y5 = np.where(df['SYS'].values < threshold, minimum_value, df['SYS'].values)

    s1 = np.where(df['sLABEL'].values < threshold, minimum_value, df['sLABEL'].values)
    s2 = np.where(df['sTop-K'].values < threshold, minimum_value, df['sTop-K'].values)
    s3 = np.where(df['sAPS'].values < threshold, minimum_value, df['sAPS'].values)
    s4 = np.where(df['sRAPS'].values < threshold, minimum_value, df['sRAPS'].values)
    s5 = np.where(df['sSYS'].values < threshold, minimum_value, df['sSYS'].values)

    b1 = np.where(df['bLABEL'].values < threshold, minimum_value, df['bLABEL'].values)
    b2 = np.where(df['bTop-K'].values < threshold, minimum_value, df['bTop-K'].values)
    b3 = np.where(df['bAPS'].values < threshold, minimum_value, df['bAPS'].values)
    b4 = np.where(df['bRAPS'].values < threshold, minimum_value, df['bRAPS'].values)
    b5 = np.where(df['bSYS'].values < threshold, minimum_value, df['bSYS'].values)

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

def drifting_vul_modified_3(path, name):
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
    # color = ['#cfb8f5', '#b08cee', '', '', '#ba9bf0', '#a67dec', '#b08cee', '#925fe8']
    # Plot the same data on both axes
    ax1.bar(x=np.arange(len(x)), height=y1, label='Design time', width=bar_width,
            edgecolor='#b08cee', linewidth=1, color='#cfb8f5', alpha=0.9, align="center")
    ax1.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
            edgecolor='#925fe8', linewidth=1, color='#b08cee', alpha=0.9,hatch='////')
    # ax1.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #         edgecolor='#261a4d', linewidth=1, color='#473192', alpha=0.9)

    ax2.bar(x=np.arange(len(x)), height=y1, label='Design time', width=bar_width,
            edgecolor='#b08cee', linewidth=1, color='#cfb8f5', alpha=0.9, align="center")
    ax2.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
            edgecolor='#925fe8', linewidth=1, color='#b08cee', alpha=0.9,hatch='////')
    # ax2.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
    #         edgecolor='#261a4d', linewidth=1, color='#473192', alpha=0.9)

    # Set the y limits
    ax1.set_ylim(0.7, 1.01)
    ax2.set_ylim(0.1, 0.2)

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
    ax1.set_yticks(np.arange(0.7, 1.01, 0.1))
    ax1.set_yticklabels([0.7, 0.8, 0.9, 1.0],
                        fontdict={'horizontalalignment': 'right', 'size': 20, 'family': 'Arial'})
    ax2.set_yticks(np.arange(0.1, 0.21, 0.1))

    ax2.set_xticks(np.arange(0, 3, 1) + 0.05)
    ax2.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20, 'family': 'Arial'})

    ax2.set_yticks(np.arange(0, 0.21, 0.1))
    ax2.set_yticklabels([0,0.1, 0.2],
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

def drifting_vul_il_3(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # y1 = df['Ori'].values
    y2 = df['Drift'].values
    y3 = df['IL'].values


    # Set up the figure and axes with custom height ratios
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 3), dpi=80, gridspec_kw={'height_ratios': [1.5, 1]})
    fig.subplots_adjust(hspace=0.001)  # adjust the space between the axes

    bar_width = 0.15

    # Plot the same data on both axes
    # color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#4b9441']
    # ax1.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
    #         edgecolor='#D3D3D3', linewidth=1, color='#ede5fb', alpha=0.9, align="center")
    ax1.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Native deployment', width=bar_width,
            edgecolor='#6bb960', linewidth=1, color='#a2d39b', alpha=0.9)
    ax1.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
            edgecolor='#397131', linewidth=1, color='#57ac4b', alpha=0.9,hatch='////')

    # ax2.bar(x=np.arange(len(x)), height=y1, label='Training', width=bar_width,
    #         edgecolor='#D3D3D3', linewidth=1, color='#ede5fb', alpha=0.9, align="center")
    ax2.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Native deployment', width=bar_width,
            edgecolor='#6bb960', linewidth=1, color='#a2d39b', alpha=0.9)
    ax2.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
            edgecolor='#397131', linewidth=1, color='#57ac4b', alpha=0.9,hatch='////')

    # Set the y limits
    ax1.set_ylim(0.8, 1.01)
    ax2.set_ylim(0.1, 0.2)

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
    ax1.set_yticks(np.arange(0.7, 1.01, 0.1))
    ax1.set_yticklabels([0.7, 0.8, 0.9, 1.0],
                        fontdict={'horizontalalignment': 'right', 'size': 20, 'family': 'Arial'})
    ax2.set_yticks(np.arange(0.1, 0.21, 0.1))

    ax2.set_xticks(np.arange(0, 3, 1) + 0.3)
    ax2.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 20, 'family': 'Arial'})

    ax2.set_yticks(np.arange(0, 0.21, 0.1))
    ax2.set_yticklabels([0,0.1, 0.2],
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
    ax1.legend(loc='center left', bbox_to_anchor=(-0.25 , 1.22), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.savefig(str(name) + '.pdf')
    plt.show()

def ae_vul_plot_script(case=''):
    drifting_vul_modified_3(r'./figures_plot/data/ae_IL_vul.xlsx',
                            r'./figures_plot/figure/drifting_vul')
    print("Figure 7(d) C4: vulnerability detection. The resulting performance when using an ML model for decision making.")


    drifting_vul(r'./figures_plot/data/ae_drifting_vul.xlsx',
                 r'./figures_plot/figure/detectdrifting_vul')

    print("Figure 8(d) C4: vulnerability detection. Prom’s performance for detecting drifting samples across case studies and underlying models.")

    drifting_vul_il_3(r'./figures_plot/data/ae_IL_vul.xlsx',
                      r'./figures_plot/figure/IL_vul')

    print("Figure 9(d) C4: vulnerability detection. Prom enhances performance through incremental learning in different underlying models.")


    Individual(r'./figures_plot/data/ae_indiv_vul.xlsx',
               r'./figures_plot/figure/individual_vul')

    print("Figure 11(d) C4: vulnerability detection. Performance of individual nonconformity functions.")

# ae_vul_plot_script()
# violin_3(method='thread')