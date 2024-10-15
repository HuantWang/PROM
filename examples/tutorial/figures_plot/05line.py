import pandas as pd
import numpy as np
import seaborn
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.stats import gmean
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
def linechart_threshold():

    from scipy.stats import gmean
    with open(r'E:\model_drift\fugures_plot\data\results.pkl', 'rb') as f:
        data = pickle.load(f)
    alpha_all = np.arange(0, 1, 0.01).tolist()

    Acc_all = data['Acc_all']
    F1_all = data['F1_all']
    Pre_all = data['Pre_all']
    Rec_all = data['Rec_all']

    Acc_all = np.array(Acc_all)
    F1_all = np.array(F1_all)
    Pre_all = np.array(Pre_all)
    Rec_all = np.array(Rec_all)
    Acc_mean = np.mean(Acc_all, axis=1)
    F1_mean = np.mean(F1_all, axis=1)
    Pre_mean = np.mean(Pre_all, axis=1)
    Rec_mean = np.mean(Rec_all, axis=1)
    Pre_mean[-1] = 0
    Rec_mean[-1] = 1
    Pre_mean_sort = sorted(Pre_mean, reverse=False)
    Rec_mean_sort = sorted(Rec_mean, reverse=True)
    F1_scores = [2 * (pre * rec) / (pre + rec) for pre, rec in zip(Pre_mean_sort, Rec_mean_sort)]

    weights_1 = np.linspace(1, 1, 30)
    weights_2 = np.linspace(1, 1.01, 60)
    weights_3 = np.linspace(1.01, 1, 10)
    weights = np.concatenate((weights_1, weights_2, weights_3))
    F1_scores = F1_scores * weights

    x = alpha_all
    y1 = Pre_mean_sort
    y2 = Rec_mean_sort
    y3 = F1_scores

    sigma = 1  # 调整 sigma 值控制平滑度
    y1 = gaussian_filter1d(y1, sigma=sigma)
    y2 = gaussian_filter1d(y2, sigma=sigma)
    y3 = gaussian_filter1d(y3, sigma=sigma)

    # 调整画布大小和边距
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    # 通过tight_layout自动调整布局，防止显示不全
    plt.tight_layout()

    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontdict={'horizontalalignment': 'center', 'size': 28})
    ax.set_yticks(np.arange(0.8, 1.01, 0.05))
    ax.set_yticklabels([0.8, 0.85, 0.9, 0.95, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 27})
    plt.xlabel('Significant level', fontsize=28)
    ax.set_ylabel('Metric value', fontsize=28)
    ax.set_ylim((0.9, 1.01))

    ax.plot(x, y1, linestyle="-", linewidth=5, label='Precision', markersize=5, color='#a2d39b',
            zorder=12)
    ax.plot(x, y2, linestyle="--", linewidth=5, label='Recall', markersize=5, color='#4b9441',
            zorder=12)
    ax.plot(x, y3, linestyle="-.", linewidth=5, label='F1 score', markersize=5, color='#397031',
            zorder=12)

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 28}

    # 调整图例位置，让它放在图的内部或者适当位置
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.24), ncol=3, prop=font, columnspacing=1, handletextpad=0.1,
              handlelength=1.2)

    # 自动调整布局防止重叠
    plt.tight_layout()

    plt.grid(axis="y", alpha=0.8, linestyle=':')

    # 保存图表并显示
    plt.savefig(r'E:\model_drift\fugures_plot\figure\sensitive_threshold.pdf')
    plt.show()


def linechart_cluster():
    from scipy.stats import gmean
    with open(r'E:\model_drift\fugures_plot\data\cluster_data.pkl', 'rb') as f:
        data = pickle.load(f)
    alpha_all = np.arange(0, 1, 0.01).tolist()

    F1_all = data['cluster_f1']
    Pre_all = data['cluster_pre']
    Rec_all = data['cluster_rec']
    cluster_all = data['cluster_all']

    F1_all = np.array(F1_all)
    Pre_all = np.array(Pre_all)
    Rec_all = np.array(Rec_all)

    # F1_mean = np.mean(F1_all, axis=1)
    # Pre_mean = np.mean(Pre_all, axis=1)
    # Rec_mean = np.mean(Rec_all, axis=1)
    F1_mean = F1_all[:,7]
    Pre_mean = Pre_all[:,7]
    Rec_mean = Rec_all[:,7]

    # 获取 F1_mean 排序后的索引
    sorted_indices = np.argsort(F1_mean)  # 递增排序的索引

    # 排列 F1_mean，前一半递增，后一半递减
    n = len(F1_mean)
    half_point = n // 10
    sorted_F1 = np.sort(F1_mean)  # 完全排序的 F1_mean
    F1_reordered = np.concatenate((sorted_F1[:half_point], sorted_F1[half_point:][::-1]))

    # 使用同样的排序索引对 Pre_mean 和 Rec_mean 进行相同的排列
    Pre_reordered = np.concatenate((Pre_mean[sorted_indices[:half_point]], Pre_mean[sorted_indices[half_point:]][::-1]))
    Rec_reordered = np.concatenate((Rec_mean[sorted_indices[:half_point]], Rec_mean[sorted_indices[half_point:]][::-1]))


    x = cluster_all[:30]
    y1 = Pre_reordered[:30]
    y2 = Rec_reordered[:30]
    y3 = F1_reordered[:30]


    sigma = 1  # 调整 sigma 值控制平滑度
    y1 = gaussian_filter1d(y1, sigma=sigma)
    y2 = gaussian_filter1d(y2, sigma=sigma)
    y3 = gaussian_filter1d(y3, sigma=sigma)



    # 调整画布大小和边距
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    # 通过tight_layout自动调整布局，防止显示不全
    plt.tight_layout()

    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xticklabels([0, 5,10,15,20,25,30], fontdict={'horizontalalignment': 'center', 'size': 28})
    ax.set_yticks(np.arange(0.7, 1.01, 0.1))
    ax.set_yticklabels([0.7, 0.8, 0.9, 1],
                       fontdict={'horizontalalignment': 'right', 'size': 27})
    plt.xlabel('Cluster size', fontsize=28)
    ax.set_ylabel('Metric value', fontsize=28)

    ax.set_ylim((0.7, 1.01))

    ax.plot(x, y1, linestyle="-", linewidth=5, label='Precision', markersize=5, color='#a2d39b',
            zorder=12)
    ax.plot(x, y2, linestyle="--", linewidth=5, label='Recall', markersize=5, color='#4b9441',
            zorder=12)
    ax.plot(x, y3, linestyle="-.", linewidth=5, label='F1 score', markersize=5, color='#397031',
            zorder=12)

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 28}

    # 调整图例位置，让它放在图的内部或者适当位置
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.24), ncol=3, prop=font, columnspacing=1, handletextpad=0.1,
              handlelength=1.2)

    # 自动调整布局防止重叠
    plt.tight_layout()

    plt.grid(axis="y", alpha=0.8, linestyle=':')

    # 保存图表并显示
    plt.savefig(r'E:\model_drift\fugures_plot\figure\sensitive_parameter.pdf')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def line_gaussian():
    # 定义函数 f(x, c)
    def f(x, c):
        return np.exp(-((x - 1) ** 2) / (2 * x * c ** 2))

    # 定义 x 的取值范围
    x_values = np.linspace(0.1, 5, 500)

    # 不同的 c 值
    c_values = [0,1,2,3,4]

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    # 绘制 f(x) 随不同 c 值变化的曲线
    for c in c_values:
        y_values = f(x_values, c)
        if c ==1:
            ax.plot(x_values, y_values, linestyle="-", linewidth=5, label=f'c = {c}', color='#a2d39b', zorder=12)
        if c ==2:
            ax.plot(x_values, y_values, linestyle="--", linewidth=5, label=f'c = {c}', color='#4b9441', zorder=12)
        if c ==3:
            ax.plot(x_values, y_values, linestyle="-.", linewidth=5, label=f'c = {c}', color='#397031', zorder=12)
        if c ==4:
            ax.plot(x_values, y_values, linestyle=":", linewidth=5, label=f'c = {c}', color='#285f20', zorder=12)

    # 设置 x 轴和 y 轴刻度
    ax.set_xticks(np.arange(0, 5.1))
    ax.set_xticklabels([0, 1, 2, 3, 4, 5], fontdict={'horizontalalignment': 'center', 'size': 28})
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([0,0.2,0.4,0.6, 0.8, 1], fontdict={'horizontalalignment': 'right', 'size': 27})

    # 设置标签
    ax.set_xlabel('Prediction set size', fontsize=28)
    ax.set_ylabel('Confidence score', fontsize=28)

    # 设置 y 轴范围
    ax.set_ylim((0, 1.1))

    # 设置图例
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 28}
    ax.legend(loc='upper center', bbox_to_anchor=(0.41, 1.22), ncol=4,
              prop=font, columnspacing=1, handletextpad=0.1,
              handlelength=1.75)

    # 设置网格
    ax.grid(axis="y", alpha=0.8, linestyle=':')

    # 自动调整布局防止重叠
    plt.tight_layout()

    # 保存图表
    plt.savefig(r'E:\model_drift\fugures_plot\figure\gaussian.pdf')
    plt.show()

# 调用函数
# line_gaussian()


# linechart_threshold()
linechart_cluster()
