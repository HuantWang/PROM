import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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
    ax1.legend(loc='center left', bbox_to_anchor=(-0.15, 1.22), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
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
        magni_train = r'E:\model_drift\fugures_plot\pkl\thread\Magni_train_0.8518134848754977_2327.pkl'
        deeptune_train = r'E:\model_drift\fugures_plot\box\deep\box_plot_deploy0.8328986364257807_8264_data.pkl'
        poem_train =  r'E:\model_drift\fugures_plot\pkl\thread\poem_train_0.88_1254.pkl'
        magni_test = r'E:\model_drift\fugures_plot\pkl\thread\poem_test_0.736010457908445_251.pkl'
        deeptune_test = r'E:\model_drift\fugures_plot\box\deep\box_plot_deploy0.5819776990305028_93_data.pkl'
        poem_test = r'E:\model_drift\fugures_plot\pkl\thread\Magini_test_0.7294382843190406_3638_data.pkl'


        # Load data from files
        with open(magni_train, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
        with open(magni_test, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
            data_distri_2 = np.concatenate((data_distri_2, np.array([0.2])))

        with open(deeptune_train, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
        with open(deeptune_test, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
        with open(poem_train, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            data_distri_5 = data_distri_5[data_distri_5 > 0.2]
        with open(poem_test, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values


    if method=='thread_il':
        color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#397131']
        magni_test = r'E:\model_drift\fugures_plot\pkl\thread\poem_test_0.736010457908445_251.pkl'
        deeptune_test = r'E:\model_drift\fugures_plot\pkl\thread\deeptune_test_0.57831126777723_2995.pkl'
        poem_test = r'E:\model_drift\fugures_plot\pkl\thread\Magini_test_0.7294382843190406_3638_data.pkl'
        magni_rtrain = r'E:\model_drift\fugures_plot\pkl\thread\magni_rt_plot_0.8268867140849601_7338.pkl'
        deeptune_rtrain = r'E:\model_drift\fugures_plot\pkl\thread\deeptune_rt_0.8315232629292073_5963.pkl'
        poem_rtrain = r'E:\model_drift\fugures_plot\pkl\thread\poem_rt_0.8407663045674678_2552.pkl'

        # Load data from files
        with open(magni_test, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
            data_distri = np.concatenate((data_distri, np.array([0.25])))
        with open(magni_rtrain, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
        with open(deeptune_test, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
        with open(deeptune_rtrain, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
        with open(poem_test, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
        with open(poem_rtrain, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values


    if method=='loop':
        color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']
        k_train = r'E:\model_drift\fugures_plot\pkl\loop\k_train_0.7022058947237987_6574.pkl'
        deeptune_train = r'E:\model_drift\fugures_plot\pkl\loop\deep_train_0.7375948160915333_5347.pkl'
        poem_train =  r'E:\model_drift\fugures_plot\pkl\loop\poem_train_0.8501353791405742_934.pkl'
        k_test = r'E:\model_drift\fugures_plot\pkl\loop\k_test_0.6360769126962933_3934.pkl'
        deeptune_test = r'E:\model_drift\fugures_plot\pkl\loop\deep_test_0.6171439421465302_725.pkl'
        poem_test = r'E:\model_drift\fugures_plot\pkl\loop\poem_test_0.7111187152668318_857.pkl'


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
        with open(poem_train, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            data_distri_5 = data_distri_5[data_distri_5 > 0.4]
        with open(poem_test, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values


    if method=='loop_il':
        color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#397131']
        k_test = r'E:\model_drift\fugures_plot\pkl\loop\k_test_0.6360769126962933_3934.pkl'
        deeptune_test = r'E:\model_drift\fugures_plot\pkl\loop\deep_test_0.6171439421465302_725.pkl'
        poem_test = r'E:\model_drift\fugures_plot\pkl\loop\poem_test_0.7111187152668318_857.pkl'
        k_rtrain = r'E:\model_drift\fugures_plot\pkl\loop\k_retrain_0.6847278411153574_3745.pkl'
        deeptune_rtrain = r'E:\model_drift\fugures_plot\pkl\loop\deep_rt_0.7001682740644295_5151.pkl'
        poem_rtrain = r'E:\model_drift\fugures_plot\pkl\loop\poem_retrain_0.7573418074582537_1435.pkl'

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
        with open(poem_test, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
        with open(poem_rtrain, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values
            data_distri_6 = data_distri_6[data_distri_6 > 0.25]



    if method=='dev':
        color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']

        b_train = r'E:\model_drift\fugures_plot\box\i2v\box_plot_train0.9000208185635654_6794_data.pkl'
        deeptune_train = r'E:\model_drift\fugures_plot\box\deep\box_plot_deploy0.8328986364257807_8264_data.pkl'
        poem_train =  r'E:\model_drift\fugures_plot\box\programl\box_plot_deploy0.8169234319726472_1150_data.pkl'
        b_test = r'E:\model_drift\fugures_plot\box\i2v\box_plot_deploy0.7131319119050349_8882_data.pkl'
        deeptune_test = r'E:\model_drift\fugures_plot\box\deep\box_plot_deploy0.5819776990305028_93_data.pkl'
        poem_test = r'E:\model_drift\fugures_plot\box\programl\box_plot_deploy0.5805412782620027_838_data.pkl'


        # Load data from files
        with open(b_train, 'rb') as file:
            data_distri = pickle.load(file)['Data'].values
            # data_distri = data_distri[data_distri > 0.25]
        with open(b_test, 'rb') as file:
            data_distri_2 = pickle.load(file)['Data'].values
        with open(deeptune_train, 'rb') as file:
            data_distri_3 = pickle.load(file)['Data'].values
            # data_distri_3 = data_distri_3[data_distri_3 > 0.3]
        with open(deeptune_test, 'rb') as file:
            data_distri_4 = pickle.load(file)['Data'].values
        with open(poem_train, 'rb') as file:
            data_distri_5 = pickle.load(file)['Data'].values
            # data_distri_5 = data_distri_5[data_distri_5 > 0.4]
        with open(poem_test, 'rb') as file:
            data_distri_6 = pickle.load(file)['Data'].values


    if method=='dev_il':
        color = ['#a2d39b', '#57ac4b', '', '', '#6bb960', '#4b9441', '#6bb960', '#397131']


        b_rtrain = r'E:\model_drift\fugures_plot\box\i2v\box_plot_IL0.8116132541229645_8882_data.pkl'
        deeptune_rtrain = r'E:\model_drift\fugures_plot\box\deep\box_plot_IL0.7786801903916476_93_data.pkl'
        poem_rtrain = r'E:\model_drift\fugures_plot\pkl\dev\il_poem_0.8376295092838051_9914.pkl'
        b_test = r'E:\model_drift\fugures_plot\box\i2v\box_plot_deploy0.7131319119050349_8882_data.pkl'
        deeptune_test = r'E:\model_drift\fugures_plot\box\deep\box_plot_deploy0.5819776990305028_93_data.pkl'
        poem_test = r'E:\model_drift\fugures_plot\box\programl\box_plot_IL0.7725996926213913_838_data.pkl'

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
        with open(poem_test, 'rb') as file:
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
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, 1.3),loc='upper center',prop=font,ncol=4,columnspacing=0.5, handletextpad=0.1,
                  handlelength=1.15)  # 你可以调整位置参数
    # ax.legend(loc='center left', bbox_to_anchor=(-0.15, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
    #               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    # Add titles, labels, and custom x-tick labels
    # plt.title('Enhanced Violin and Box Plot of Six Datasets', fontsize=16)
    # plt.xlabel('Method', fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    if method == 'thread_il':
        plt.xticks([1.5, 4, 6.5], ['Magni', 'DeepTune', 'POEM'], fontsize=16)
    elif method == 'thread':
        plt.xticks([1.5, 4, 6.5], ['Magni', 'DeepTune', 'POEM'], fontsize=16)
    elif method == 'loop':
        plt.xticks([1.5, 4, 6.5], ['K.Stock et al.', 'DeepTune', 'POEM'], fontsize=16)
    elif method == 'loop_il':
        plt.xticks([1.5, 4, 6.5], ['K.Stock et al.', 'DeepTune', 'POEM'], fontsize=16)
    elif method == 'dev_il':
        plt.xticks([1.5, 4, 6.5], ['DeepTune', 'IR2Vec', 'Programl'], fontsize=16)
    elif method == 'dev':
        plt.xticks([1.5, 4, 6.5], ['DeepTune', 'IR2Vec', 'Programl'], fontsize=16)
    plt.ylabel('Perf. to the oracle', fontdict={'horizontalalignment': 'center', 'size': 15,'family': 'Arial'})
    # plt.xticks([1.5, 4, 6.5], ['Magni', 'DeepTune', 'POEM'], fontsize=14)

    # Add grid
    plt.grid(True)
    plt.savefig(r'E:\model_drift\fugures_plot\figure\\' + method + '.pdf')
    # Show the plot
    plt.show()

def violin_4():
    color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']

    mlp_test = r'E:\model_drift\fugures_plot\pkl\split\deep_test_0.6171439421465302_725.pkl'
    svm_test = r'E:\model_drift\fugures_plot\pkl\split\svm_test_0.6529881177167827_2876_data.pkl'
    lstm_test = r'E:\model_drift\fugures_plot\pkl\split\deep_test_0.6171439421465302_725.pkl'
    gnn_test = r'E:\model_drift\fugures_plot\pkl\split\poem_test_0.7111187152668318_857.pkl'
    mlp_train = r'E:\model_drift\fugures_plot\pkl\split\k_retrain_0.6847278411153574_3745.pkl'
    svm_train = r'E:\model_drift\fugures_plot\pkl\split\svm_train_0.7058824844470889_1449_data.pkl'
    lstm_train = r'E:\model_drift\fugures_plot\pkl\split\deep_rt_0.7001682740644295_5151.pkl'
    gnn_train = r'E:\model_drift\fugures_plot\pkl\split\poem_retrain_0.7573418074582537_1435.pkl'

    # Load data from files
    with open(mlp_test, 'rb') as file:
        data_distri_3 = pickle.load(file)['Data'].values
    with open(mlp_train, 'rb') as file:
        data_distri_4 = pickle.load(file)['Data'].values
        data_distri_4 = data_distri_4[data_distri_4 > 0.22]
    with open(svm_test, 'rb') as file:
        data_distri = pickle.load(file)['Data'].values
    with open(svm_train, 'rb') as file:
        data_distri_2 = pickle.load(file)['Data'].values
        data_distri_2 = data_distri_2[data_distri_2 > 0.1]
    with open(lstm_test, 'rb') as file:
        data_distri_5 = pickle.load(file)['Data'].values
    with open(lstm_train, 'rb') as file:
        data_distri_6 = pickle.load(file)['Data'].values
        data_distri_6 = data_distri_6[data_distri_6 > 0.22]
    with open(gnn_test, 'rb') as file:
        data_distri_7 = pickle.load(file)['Data'].values
    with open(gnn_train, 'rb') as file:
        data_distri_8 = pickle.load(file)['Data'].values
        data_distri_8 = data_distri_8[data_distri_8 > 0.25]

    # Prepare data for plotting
    data_for_plotting = [data_distri, data_distri_2, data_distri_3, data_distri_4, data_distri_5, data_distri_6,
                         data_distri_7, data_distri_8]


    # Plot aesthetics settings
    figsize = (6, 3)
    box_widths = 0.2
    violin_widths = 1
    # colors_violin = ['#e3d6f9', '#c5aaf2', '#e3d6f9', '#b7a5d8', '#e3d6f9', '#b7a5d8','#e3d6f9', '#b7a5d8']
    # colors_box = ['#cfb8f5', '#a67dec', '#cfb8f5', '#a67dec', '#cfb8f5', '#a67dec', '#cfb8f5', '#a67dec']
    # edge_color_violin = ['#ba9bf0', '#9c6eea', '#ba9bf0', '#9c6eea', '#ba9bf0', '#9c6eea','#ba9bf0', '#9c6eea']
    # edge_color_box = ['#b08cee', '#9c6eea', '#b08cee', '#9c6eea', '#b08cee', '#9c6eea','#b08cee', '#9c6eea']
    colors_violin = [color[0], color[1], color[0], color[1], color[0], color[1], color[0], color[1]]
    colors_box = [color[0], color[1], color[0], color[1], color[0], color[1], color[0], color[1]]
    edge_color_violin = [color[4], color[5], color[4], color[5], color[4], color[5], color[4], color[5]]
    edge_color_box = [color[6], color[7], color[6], color[7], color[6], color[7], color[6], color[7]]
    # 为每种数据集创建一个图例标签
    legend_labels = ['Native deployment', 'PROM on Deployment']
    legend_colors = [color[0],color[1]]  # 相应的颜色


    fig = plt.figure(figsize=figsize, dpi=80)
    ax = fig.add_axes([0.15, 0.2, 0.8, 0.6], zorder=11)
    ax.set_yticklabels([-0.1,0,0.2,0.4,0.6,0.8,1],
                           fontdict={'horizontalalignment': 'right', 'size': 14})
    # Create violin plot
    parts = plt.violinplot(data_for_plotting, positions=[1, 2, 4, 5, 7, 8,10,11],
                           showmeans=False, showmedians=True,
                           showextrema=True, widths=violin_widths)

    # Create box plot
    box = plt.boxplot(data_for_plotting, positions=[1, 2, 4, 5, 7, 8,10,11], widths=box_widths, patch_artist=True)

    # Customize colors and styles
    set_box_colors(box, colors_box,edge_color_box)
    set_violin_colors(parts, colors_violin, edge_color_violin)

    # Customize the parts of the violin plot


    font = {'family':'Arial',
                'weight': 'light',
                'size': 16
                }


    # 创建代表每个数据集的矩形图形
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(legend_labels, legend_colors)]

    # 在图中添加图例
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, 1.3),loc='upper center',prop=font,ncol=4,columnspacing=0.5, handletextpad=0.1,
                  handlelength=1.15)  # 你可以调整位置参数
    # ax.legend(loc='center left', bbox_to_anchor=(-0.15, 1.2), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
    #               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    # Add titles, labels, and custom x-tick labels
    # plt.title('Enhanced Violin and Box Plot of Six Datasets', fontsize=16)
    # plt.xlabel('Method', fontdict={'horizontalalignment': 'center', 'size': 20,'family': 'Arial'})
    plt.ylabel('Perf. to the oracle', fontdict={'horizontalalignment': 'center', 'size': 15,'family': 'Arial'})
    plt.xticks([1.5, 4.5, 7.5,10.5], ['SVM', 'MLP', 'LSTM','GNN'], fontsize=16)

    # Add grid
    plt.grid(True)
    plt.savefig(r'E:\model_drift\fugures_plot\figure\drifting_CASE2' + '.pdf')
    # Show the plot
    plt.show()

def drifting_vul_il(path, name):
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

    ax2.set_xticks(np.arange(0, 4, 1) + 0.3)
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
    ax1.legend(loc='center left', bbox_to_anchor=(-0.05, 1.22), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.savefig(str(name) + '.pdf')
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


def set_box_colors(box, colors_box, edge_color_box):
    for patch, color, edge_color in zip(box['boxes'], colors_box, edge_color_box):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge_color)
    for whisker, edge_color in zip(box['whiskers'], edge_color_box * 2):
        whisker.set_color(edge_color)
    for cap, edge_color in zip(box['caps'], edge_color_box * 2):
        cap.set_color(edge_color)
    for median in box['medians']:
        median.set_color('black')


def violin_4_score():
    color = ['#e3d6f9', '#c5aaf2', '', '', '#ba9bf0', '#9c6eea', '#b08cee', '#9c6eea']

    mlp_test = r'E:\model_drift\fugures_plot\pkl\split\deep_test_0.6171439421465302_725.pkl'
    svm_test = r'E:\model_drift\fugures_plot\pkl\split\svm_test_0.6529881177167827_2876_data.pkl'
    lstm_test = r'E:\model_drift\fugures_plot\pkl\split\deep_test_0.6171439421465302_725.pkl'
    gnn_test = r'E:\model_drift\fugures_plot\pkl\split\poem_test_0.7111187152668318_857.pkl'
    mlp_train = r'E:\model_drift\fugures_plot\pkl\split\k_retrain_0.6847278411153574_3745.pkl'
    svm_train = r'E:\model_drift\fugures_plot\pkl\split\svm_train_0.7058824844470889_1449_data.pkl'
    lstm_train = r'E:\model_drift\fugures_plot\pkl\split\deep_rt_0.7001682740644295_5151.pkl'
    gnn_train = r'E:\model_drift\fugures_plot\pkl\split\poem_retrain_0.7573418074582537_1435.pkl'

    # Load data from files
    with open(mlp_test, 'rb') as file:
        data_distri_3 = pickle.load(file)['Data'].values
    with open(mlp_train, 'rb') as file:
        data_distri_4 = pickle.load(file)['Data'].values
        data_distri_4 = data_distri_4[data_distri_4 > 0.22]
    with open(svm_test, 'rb') as file:
        data_distri = pickle.load(file)['Data'].values
    with open(svm_train, 'rb') as file:
        data_distri_2 = pickle.load(file)['Data'].values
        data_distri_2 = data_distri_2[data_distri_2 > 0.1]
    with open(lstm_test, 'rb') as file:
        data_distri_5 = pickle.load(file)['Data'].values
    with open(lstm_train, 'rb') as file:
        data_distri_6 = pickle.load(file)['Data'].values
        data_distri_6 = data_distri_6[data_distri_6 > 0.22]
    with open(gnn_test, 'rb') as file:
        data_distri_7 = pickle.load(file)['Data'].values
    with open(gnn_train, 'rb') as file:
        data_distri_8 = pickle.load(file)['Data'].values
        data_distri_8 = data_distri_8[data_distri_8 > 0.25]

    # Prepare data for plotting
    data_for_plotting = [data_distri, data_distri_2, data_distri_3, data_distri_4, data_distri_5, data_distri_6,
                         data_distri_7, data_distri_8]
    data_for_plotting_wrong = [data_distri, data_distri_2, data_distri_3, data_distri_4, data_distri_5, data_distri_6,
                               data_distri_7, data_distri_8]

    # Plot aesthetics settings
    figsize = (6, 3)
    box_widths = 0.3
    colors_box = [color[0], color[1], color[0], color[1], color[0], color[1], color[0], color[1]]
    edge_color_box = [color[6], color[7], color[6], color[7], color[6], color[7], color[6], color[7]]

    # 为每种数据集创建一个图例标签
    legend_labels = ['Native deployment', 'PROM on Deployment']
    legend_colors = [color[0], color[1]]  # 相应的颜色

    fig = plt.figure(figsize=figsize, dpi=80)
    ax = fig.add_axes([0.15, 0.2, 0.8, 0.6], zorder=11)
    ax.set_yticklabels([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 14})

    # Create box plot for data_for_plotting
    box = plt.boxplot(data_for_plotting, positions=[1, 2, 4, 5, 7, 8, 10, 11], widths=box_widths, patch_artist=True)
    set_box_colors(box, colors_box, edge_color_box)

    # Create box plot for data_for_plotting_wrong
    box_wrong = plt.boxplot(data_for_plotting_wrong, positions=[1.5, 2.5, 4.5, 5.5, 7.5, 8.5, 10.5, 11.5],
                            widths=box_widths, patch_artist=True)
    set_box_colors(box_wrong, colors_box, edge_color_box)

    font = {'family': 'Arial',
            'weight': 'light',
            'size': 16}

    # 创建代表每个数据集的矩形图形
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(legend_labels, legend_colors)]

    # 在图中添加图例
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, 1.3), loc='upper center', prop=font, ncol=4,
               columnspacing=0.5, handletextpad=0.1,
               handlelength=1.15)  # 你可以调整位置参数

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.ylabel('Perf. to the oracle', fontdict={'horizontalalignment': 'center', 'size': 15, 'family': 'Arial'})
    plt.xticks([1.5, 4.5, 7.5, 10.5], ['SVM', 'MLP', 'LSTM', 'GNN'], fontsize=16)

    plt.grid(True)
    plt.savefig(r'E:\model_drift\fugures_plot\figure\drifting_CASE2' + '.pdf')
    plt.show()


# Run the function to generate the plot
# violin_4_score()

#三个方法的
# violin_4()
# violin_3(method='thread')
# violin_3(method='thread_il')
# violin_3(method='loop')
# violin_3(method='loop_il')
# violin_3(method='dev')
# violin_3(method='dev_il')
# drifting_vul_modified_3(r'E:\model_drift\fugures_plot\data\IL_vul.xlsx',r'E:\model_drift\fugures_plot\figure\drifting_vul')
# drifting_vul_il_3(r'E:\model_drift\fugures_plot\data\IL_vul.xlsx',r'E:\model_drift\fugures_plot\figure\IL_vul')