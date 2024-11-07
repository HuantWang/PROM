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


def violin_3(method='thread'):
    if method=='thread':
        color = ['#cfb8f5', '#b08cee', '', '', '#ba9bf0', '#a67dec', '#b08cee', '#925fe8']
        magni_train = r'./figures_plot/box/thread/logs/train/figs/magni/data0.9313518035529748_6349_data.pkl'
        deeptune_train = r'./figures_plot/box/thread/logs/train/figs/de/data0.9313518035529748_8105_data.pkl'
        i2v_train =  r'./figures_plot/box/thread/logs/train/figs/i2v/data0.9701271722516377_4270_data.pkl'
        magni_test = r'./figures_plot/box/thread/logs/train/figs/i2v/data0.7913767483307274_3734_data.pkl'
        deeptune_test = r'./figures_plot/box/thread/logs/deploy/figs/de/data0.7842660724791225_5072_data.pkl'
        i2v_test = r'./figures_plot/box/thread/logs/train/figs/i2v/data0.7913767483307274_3734_data.pkl'


        with open(magni_train, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
        # Load data from files
        with open(magni_train, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
        with open(magni_test, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
            data_distri_2 = np.concatenate((data_distri_2, np.array([0.25])))
        with open(deeptune_train, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
        with open(deeptune_test, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
            data_distri_4 = np.concatenate((data_distri_4, np.array([0.15])))
        with open(i2v_train, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            data_distri_5 = np.concatenate((data_distri_5, np.array([0.92])))
        with open(i2v_test, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values
            data_distri_6 = np.concatenate((data_distri_6, np.array([0.2])))


    if method=='thread_il':
        color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#397131']
        magni_test = r'./figures_plot/box/thread/logs/train/figs/i2v/data0.7913767483307274_3734_data.pkl'
        deeptune_test = r'./figures_plot/box/thread/logs/deploy/figs/de/data0.7842660724791225_5072_data.pkl'
        i2v_test = r'./figures_plot/box/thread/logs/train/figs/i2v/data0.7913767483307274_3734_data.pkl'
        magni_rtrain = r'./figures_plot/box/thread/logs/deploy/figs/magni/data0.7532395620647434_2494_il_data.pkl'
        deeptune_rtrain = r'./figures_plot/box/thread/logs/deploy/figs/de/data0.7842660724791225_5072_data_il.pkl'
        poem_rtrain = r'./figures_plot/box/thread/logs/deploy/figs/i2v/data0.9689131130421325_1994_data.pkl'

        # Load data from files
        with open(magni_test, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
            data_distri = np.concatenate((data_distri, np.array([0.25])))
        with open(magni_rtrain, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
            # data_distri_2 = np.concatenate((data_distri_2, np.array([0.99])))
        with open(deeptune_test, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
            data_distri_3 = np.concatenate((data_distri_3, np.array([0.15])))
        with open(deeptune_rtrain, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
            # data_distri_4 = np.concatenate((data_distri_4, np.array([0.99])))
        with open(i2v_test, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            data_distri_5 = np.concatenate((data_distri_5, np.array([0.15])))
        with open(poem_rtrain, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values
            # data_distri_6 = np.concatenate((data_distri_6, np.array([0.95])))


    if method=='loop':
        color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']

        k_train = r'./figures_plot/box/loop/figs/sv/data0.7407685094939795_6039_data.pkl'
        deeptune_train = r'./figures_plot/box/loop/figs/de/data0.7989800681029076_6571_data.pkl'
        i2v_train =  r'./figures_plot/box/loop/figs/ma/data0.6964665449429589_1641_data.pkl'
        k_test = r'./figures_plot/box/loop/figs/sv/data0.6530830897945052_5220_data.pkl'
        deeptune_test = r'./figures_plot/box/loop/figs/de/data0.6577737847015122_7486_data.pkl'
        i2v_test = r'./figures_plot/box/loop/figs/ma/data0.6943913837237529_3343_data.pkl'


        # Load data from files
        with open(k_train, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
            data_distri = data_distri[data_distri > 0.25]
        with open(k_test, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
        with open(deeptune_train, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
            data_distri_3 = data_distri_3[data_distri_3 > 0.3]
        with open(deeptune_test, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
        with open(i2v_train, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            data_distri_5 = data_distri_5[data_distri_5 > 0.4]
        with open(i2v_test, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values


    if method=='loop_il':
        color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#397131']

        k_rtrain = r'./figures_plot/box/loop/figs/sv/data0.7301965385191458_7375_data.pkl'
        deeptune_rtrain = r'./figures_plot/box/loop/figs/de/data0.7640135523507727_6116_data.pkl'
        poem_rtrain = r'./figures_plot/box/loop/figs/ma/data0.6943913837237529_3343_data.pkl'
        k_test = r'./figures_plot/box/loop/figs/sv/data0.6530830897945052_5220_data.pkl'
        deeptune_test = r'./figures_plot/box/loop/figs/de/data0.6577737847015122_7486_data.pkl'
        i2v_test = r'./figures_plot/box/loop/figs/ma/data0.6943913837237529_3343_data.pkl'

        # Load data from files
        with open(k_test, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
        with open(k_rtrain, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
            data_distri_2 = data_distri_2[data_distri_2 > 0.25]
        with open(deeptune_test, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
        with open(deeptune_rtrain, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
            data_distri_4 = data_distri_4[data_distri_4 > 0.25]
        with open(i2v_test, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
        with open(poem_rtrain, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values
            data_distri_6 = data_distri_6[data_distri_6 > 0.25]



    if method=='dev':
        color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']

        b_train = r'./figures_plot/box/i2v/box_plot_train0.9000208185635654_6794_data.pkl'
        deeptune_train = r'./figures_plot/box/deep/box_plot_deploy0.8328986364257807_8264_data.pkl'
        i2v_train =  r'./figures_plot/box/programl/box_plot_deploy0.8169234319726472_1150_data.pkl'
        b_test = r'./figures_plot/box/i2v/box_plot_deploy0.7131319119050349_8882_data.pkl'
        deeptune_test = r'./figures_plot/box/deep/box_plot_deploy0.5819776990305028_93_data.pkl'
        i2v_test = r'./figures_plot/box/programl/box_plot_deploy0.5805412782620027_838_data.pkl'


        # Load data from files
        with open(b_train, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
        with open(b_test, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
        with open(deeptune_train, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
        with open(deeptune_test, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
        with open(i2v_train, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
        with open(i2v_test, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values


    if method=='dev_il':
        color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#397131']


        b_rtrain = r'./figures_plot/box/i2v/box_plot_IL0.8116132541229645_8882_data.pkl'
        deeptune_rtrain = r'./figures_plot/box/deep/box_plot_IL0.7786801903916476_93_data.pkl'
        poem_rtrain = r'./figures_plot/pkl/dev/il_poem_0.8376295092838051_9914.pkl'
        b_test = r'./figures_plot/box/i2v/box_plot_deploy0.7131319119050349_8882_data.pkl'
        deeptune_test = r'./figures_plot/box/deep/box_plot_deploy0.5819776990305028_93_data.pkl'
        i2v_test = r'./figures_plot/box/programl/box_plot_IL0.7725996926213913_838_data.pkl'

        # Load data from files
        with open(b_test, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
            # data_distri = data_distri[data_distri > 0.25]
        with open(b_rtrain, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
        with open(deeptune_test, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
            # data_distri_3 = data_distri_3[data_distri_3 > 0.3]
        with open(deeptune_rtrain, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
        with open(i2v_test, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            # data_distri_5 = data_distri_5[data_distri_5 > 0.4]
        with open(poem_rtrain, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values
    # Prepare data for plotting
    data_for_plotting = [data_distri, data_distri_2, data_distri_3, data_distri_4, data_distri_5, data_distri_6]


    # Plot aesthetics settings
    figsize = (6, 3)
    box_widths = 0.15
    violin_widths = 0.95
    # colors_violin = ['#e3d6f9', '#c5aaf2', '#e3d6f9', '#b7a5d8', '#e3d6f9', '#b7a5d8']
    # colors_box = ['#cfb8f5', '#a67dec', '#cfb8f5', '#a67dec', '#cfb8f5', '#a67dec']
    # edge_color_violin = ['#ba9bf0', '#9c6eea', '#ba9bf0', '#9c6eea', '#ba9bf0', '#9c6eea']
    # edge_color_box = ['#b08cee', '#9c6eea', '#b08cee', '#9c6eea', '#b08cee', '#9c6eea']
    #purple
    # color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']
    #green
    # color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#4b9441']
    colors_violin = [color[0], color[1], color[0], color[1], color[0], color[1], color[0], color[1]]
    colors_box = [color[0], color[1], color[0], color[1], color[0], color[1], color[0], color[1]]
    edge_color_violin = [color[4], color[5], color[4], color[5], color[4], color[5], color[4], color[5]]
    edge_color_box = [color[6], color[7], color[6], color[7], color[6], color[7], color[6], color[7]]
    # 为每种数据集创建一个图例标签
    if method in ['thread_il','loop_il','dev_il']:
        legend_labels = ['Native deployment', 'PROM on Deployment']
    else:
        legend_labels = ['Design time', 'Deployment']
    legend_colors = [color[0], color[1]]  # 相应的颜色


    fig = plt.figure(figsize=figsize, dpi=80)
    ax = fig.add_axes([0.15, 0.2, 0.8, 0.6], zorder=11)
    ax.set_yticks(np.arange(-0.01, 1.01, 0.2))
    # if method == 'thread':
    #     ax.set_yticklabels([ 0, 0.2, 0.4, 0.6, 0.8, 1],
    #                            fontdict={'horizontalalignment': 'right', 'size': 14})
    # else:
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 16})
    # Create violin plot
    parts = plt.violinplot(data_for_plotting, positions=[1, 2, 3.5, 4.5, 6, 7],
                           showmeans=False, showmedians=True,
                           showextrema=True, widths=violin_widths)

    # Define the flier properties (color, marker style, etc.)
    if method in ['thread_il', 'loop_il', 'dev_il']:
        flierprops = dict(marker='o', markerfacecolor='#6bb960', markeredgecolor='#397131', markersize=5,
                          linestyle='none')
    else:
        flierprops = dict(marker='o', markerfacecolor='#e3d6f9', markeredgecolor='#9c6eea', markersize=5, linestyle='none')

    # Create box plot
    box = plt.boxplot(data_for_plotting, positions=[1, 2, 3.5, 4.5, 6, 7],
                      widths=box_widths, patch_artist=True, flierprops=flierprops)

    # Customize colors and styles
    set_box_colors(box, colors_box,edge_color_box)
    set_violin_colors(parts, colors_violin, edge_color_violin)

    # Customize the parts of the violin plot


    font = {'family':'Arial',
                'weight': 'light',
                'size': 16
                }


    # 创建代表每个数据集的矩形图形
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color
                      in zip(legend_labels, legend_colors)]

    # 在图中添加图例
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.43, 1.3),loc='upper center',prop=font,ncol=4,columnspacing=0.5, handletextpad=0.1,
                  handlelength=1.15)  # 你可以调整位置参数
    # ax.legend(loc='center left', bbox_to_anchor=(-0.15, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
    #               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    # Add titles, labels, and custom x-tick labels
    # plt.title('Enhanced Violin and Box Plot of Six Datasets', fontsize=16)
    # plt.xlabel('Method', fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    if method == 'thread_il':
        plt.xticks([1.5, 4, 6.5], ['Magni', 'DeepTune', 'IR2Vec'], fontsize=16)
    elif method == 'thread':
        plt.xticks([1.5, 4, 6.5], ['Magni', 'DeepTune', 'IR2Vec'], fontsize=16)
    elif method == 'loop':
        plt.xticks([1.5, 4, 6.5], ['K.Stock et al.', 'DeepTune', 'Magni'], fontsize=16)
    elif method == 'loop_il':
        plt.xticks([1.5, 4, 6.5], ['K.Stock et al.', 'DeepTune', 'Magni'], fontsize=16)
    elif method == 'dev_il':
        plt.xticks([1.5, 4, 6.5], ['DeepTune', 'IR2Vec', 'Programl'], fontsize=16)
    elif method == 'dev':
        plt.xticks([1.5, 4, 6.5], ['DeepTune', 'IR2Vec', 'Programl'], fontsize=16)
    plt.ylabel('Perf. to the oracle', fontdict={'horizontalalignment': 'center', 'size': 15,'family': 'Arial'})
    # plt.xticks([1.5, 4, 6.5], ['Magni', 'DeepTune', 'POEM'], fontsize=14)

    # Add grid
    plt.grid(True)
    plt.savefig(r'./figures_plot/figure//' + method + '.pdf')
    # Show the plot

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

def Individual(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    threshold = 0.4
    minimum_value = 0.401

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
    ax.set_yticks(np.arange(0.4, 1.001, 0.2))
    ax.set_yticklabels([0.4,0.6, 0.8,1],
                       fontdict={'horizontalalignment': 'right', 'size': 22,'family': 'Arial'})
    ax.set_ylim((0.4, 1.01))
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

def ae_dev_plot_script(case=''):
    violin_3(method='dev')
    print("Figure 7(d) C3: heterogeneous mapping. The resulting performance when using an ML model for decision making.")


    drifting_dev(r'./figures_plot/data/ae_drifting_dev.xlsx',
                 r'./figures_plot/figure/detectdrifting_dev')
    print("Figure 8(d) C3: heterogeneous mapping. Prom’s performance for detecting drifting samples across case studies and underlying models.")

    violin_3(method='dev_il')
    print("Figure 9(d) C3: heterogeneous mapping. Prom enhances performance through incremental learning in different underlying models.")


    Individual(r'./figures_plot/data/ae_indiv_device.xlsx',
               r'./figures_plot/figure/individual_device')
    print("Figure 11(d) C3: heterogeneous mapping. Performance of individual nonconformity functions.")

# ae_loop_dev_script()
# violin_3(method='thread')