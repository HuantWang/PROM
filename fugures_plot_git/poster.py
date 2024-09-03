import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def drifting_vul_modified(path, name):
    df = pd.read_excel(path)
    x = np.arange(1, 6)  # 设置 x 轴的值为从 1 到 5
    y2 = df['Drift'].values
    y3 = df['IL'].values

    # Set up the figure and axes with custom height ratios
    fig, ax1 = plt.subplots(figsize=(7, 3), dpi=80)
    fig.subplots_adjust(hspace=0.001)  # 调整子图之间的间距

    bar_width = 0.15

    # 绘制数据
    ax1.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Deployment', width=bar_width,
            edgecolor='#34551d', linewidth=1, color='#abc698', alpha=0.9)
    ax1.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='PROM on Deployment', width=bar_width,
            edgecolor='#34551d', linewidth=1, color='#588e31', alpha=0.9)

    # 设置 y 轴的范围
    ax1.set_ylim(0.8, 1.01)

    # 设置 x 轴的刻度和标签
    # ax1.set_xticks(np.arange(len(x)) + bar_width + 0.09)
    ax1.set_xticks(np.arange(0, 5, 1) + 0.3)
    # ax1.set_xticklabels([f'c{i}' for i in range(1, 6)], rotation=45, ha='right', fontsize=12)

    ax1.set_xticklabels([f'C{i}' for i in range(1, 6)], fontdict={'horizontalalignment': 'center', 'size': 20, 'family': 'Arial'})
    # 设置 y 轴的刻度和标签
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontdict={'horizontalalignment': 'right', 'size': 20, 'family': 'Arial'})

    # 设置 y 轴标签的位置
    y_label = ax1.set_ylabel('Perf. to the oracle', fontsize=20)
    y_label.set_position((y_label.get_position()[0], y_label.get_position()[1] ))

    # 图例
    box = ax1.get_position()
    ax1.set_position([box.x0 + 0.05, box.y0 + 0.02, box.width, box.height * 0.8])
    font = {'family': 'Arial', 'weight': 'light', 'size': 20}
    ax1.legend(loc='center left', bbox_to_anchor=(-0.05, 1.22), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.savefig(str(name) + '.pdf')
    plt.show()


def drifting_vul_modified_1(path, name):
    df = pd.read_excel(path)
    x = np.arange(1, 6)  # 设置 x 轴的值为从 1 到 5
    y2 = df['Ori'].values
    y3 = df['Drift'].values

    # Set up the figure and axes with custom height ratios
    fig, ax1 = plt.subplots(figsize=(7, 3), dpi=80)
    fig.subplots_adjust(hspace=0.001)  # 调整子图之间的间距

    bar_width = 0.15

    # 绘制数据
    ax1.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Design time', width=bar_width,
            edgecolor='#34551d', linewidth=1, color='#abc698', alpha=0.9)
    ax1.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='Deployment', width=bar_width,
            edgecolor='#34551d', linewidth=1, color='#588e31', alpha=0.9)

    # 设置 y 轴的范围
    ax1.set_ylim(0.8, 1.01)

    # 设置 x 轴的刻度和标签
    # ax1.set_xticks(np.arange(len(x)) + bar_width + 0.09)
    ax1.set_xticks(np.arange(0, 5, 1) + 0.3)
    # ax1.set_xticklabels([f'c{i}' for i in range(1, 6)], rotation=45, ha='right', fontsize=12)

    ax1.set_xticklabels([f'C{i}' for i in range(1, 6)], fontdict={'horizontalalignment': 'center', 'size': 20, 'family': 'Arial'})
    # 设置 y 轴的刻度和标签
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontdict={'horizontalalignment': 'right', 'size': 20, 'family': 'Arial'})

    # 设置 y 轴标签的位置
    y_label = ax1.set_ylabel('Perf. to the oracle', fontsize=20)
    y_label.set_position((y_label.get_position()[0], y_label.get_position()[1] ))

    # 图例
    box = ax1.get_position()
    ax1.set_position([box.x0  , box.y0 + 0.02, box.width, box.height * 0.8])
    font = {'family': 'Arial', 'weight': 'light', 'size': 20}
    ax1.legend(loc='center left', bbox_to_anchor=(0.1, 1.22), ncol=4, prop=font, columnspacing=0.5, handletextpad=0.1,
               handlelength=1.15)

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.savefig(str(name) + '.pdf')
    plt.show()





drifting_vul_modified(r'E:\model_drift\fugures_plot\data\poster_1.xlsx',r'E:\model_drift\fugures_plot\figure\poster_1.pdf')
drifting_vul_modified_1(r'E:\model_drift\fugures_plot\data\poster_1.xlsx',r'E:\model_drift\fugures_plot\figure\poster_0.pdf')