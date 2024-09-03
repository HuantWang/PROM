import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def motivation(path,name):
    #读取数据
    df = pd.read_excel(path)
    x = df['years'].values
    # y1 = df['V_precision'].values
    # y2 = df['V_recall'].values
    y3 = df['V_f1'].values
    # y4 = df['F_precision'].values
    # y5 = df['F_recall'].values
    y6 = df['F_f1'].values

    #设置画布大小以及x轴和y轴
    fig = plt.figure(figsize=(12, 4), dpi=80)
    ax = fig.add_axes([0.12, 0.24, 0.7, 0.6], zorder=11)
    ax.set_xticks(np.arange(1,6,1))
    ax.set_xticklabels(['12-14','15-17','18-19','20-21','22-23'], fontdict={'horizontalalignment': 'center', 'size': 22})
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_yticklabels([0.2,0.4,0.6,0.8,1],
                       fontdict={'horizontalalignment': 'right', 'size': 22})
    plt.xlabel('Years', fontsize=22)

    ax.set_ylim((0.1,0.91))

    #画图
    #linestyle：'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # ax.plot(x, y1, linestyle=":",linewidth=5,label='Vulde_Pr',marker='^',markersize=18,color=seaborn.xkcd_rgb['warm grey'],zorder=12)
    # ax.plot(x, y2, linestyle=":", linewidth=5, label='Vulde_Re', marker='p', markersize=18,
    #         color=seaborn.xkcd_rgb['grey blue'], zorder=12)
    ax.plot(x, y3, linestyle="-", linewidth=4, label='Vulde', marker='X', markersize=15,
            color='#0589d2', zorder=12)
    # ax.plot(x, y4, linestyle="-",linewidth=5, label='Funded_Pr', marker='$o$', markersize=22,color=seaborn.xkcd_rgb['wisteria'],zorder=12)
    # ax.plot(x, y5, linestyle="-",linewidth=5, label='Funded_Re', marker='*', markersize=22,color=seaborn.xkcd_rgb['tan'],zorder=12)
    # ax.plot(x, y6, linestyle="-", linewidth=5, label='Funded', marker='D', markersize=14, color=seaborn.xkcd_rgb['flat green'],zorder=12)

    # 设置图例
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 28       }



    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # ax.legend(loc='center left', bbox_to_anchor=(0.22, 0.20), ncol=2, prop=font,columnspacing=0.8)
    ax.legend(loc='best', ncol=2, prop=font,columnspacing=1,handletextpad=0.1,handlelength=1.2)

    ax.set_ylabel('F1 score', fontsize=22)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    # 对保存的图片进行显示
    plt.show()

# def motivation(path,name):
#     #读取数据
#     df = pd.read_excel(path)
#     x = df['years'].values
#     # y1 = df['V_precision'].values
#     # y2 = df['V_recall'].values
#     y3 = df['V_f1'].values
#     # y4 = df['F_precision'].values
#     # y5 = df['F_recall'].values
#     y6 = df['F_f1'].values
#
#     #设置画布大小以及x轴和y轴
#     fig = plt.figure(figsize=(11, 10), dpi=80)
#     ax = fig.add_axes([0.12, 0.24, 0.7, 0.6], zorder=11)
#     # ax.set_xticks(np.arange(1,3,1))
#     ax.set_xticks([1.2,1.7])
#     ax.set_xticklabels(['2013-2015','2021-2023'], fontdict={'horizontalalignment': 'center', 'size': 28})
#     ax.set_yticks(np.arange(0.2, 1.01, 0.2))
#     ax.set_yticklabels([0.2,0.4,0.6,0.8,1],
#                        fontdict={'horizontalalignment': 'right', 'size': 27})
#     plt.xlabel('Years', fontsize=28)
#
#     ax.set_ylim((0.1,1.01))
#
#     #画图
#     #linestyle：'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
#     # ax.plot(x, y1, linestyle=":",linewidth=5,label='Vulde_Pr',marker='^',markersize=18,color=seaborn.xkcd_rgb['warm grey'],zorder=12)
#     # ax.plot(x, y2, linestyle=":", linewidth=5, label='Vulde_Re', marker='p', markersize=18,
#     #         color=seaborn.xkcd_rgb['grey blue'], zorder=12)
#     ax.plot([1.1,1.8], y3, linestyle=":", linewidth=5, label='Vulde', marker='X', markersize=20,
#             color=seaborn.xkcd_rgb['dirty pink'], zorder=12)
#     # ax.plot(x, y4, linestyle="-",linewidth=5, label='Funded_Pr', marker='$o$', markersize=22,color=seaborn.xkcd_rgb['wisteria'],zorder=12)
#     # ax.plot(x, y5, linestyle="-",linewidth=5, label='Funded_Re', marker='*', markersize=22,color=seaborn.xkcd_rgb['tan'],zorder=12)
#     ax.plot([1.1,1.8], y6, linestyle="-", linewidth=5, label='Funded', marker='D', markersize=14, color=seaborn.xkcd_rgb['flat green'],zorder=12)
#
#     # 设置图例
#     font = {'family': 'Times New Roman',
#             'weight': 'normal',
#             'size': 28       }
#
#
#
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
#     # ax.legend(loc='center left', bbox_to_anchor=(0.22, 0.20), ncol=2, prop=font,columnspacing=0.8)
#     ax.legend(loc='best', ncol=2, prop=font,columnspacing=1,handletextpad=0.1,handlelength=1.2)
#
#     ax.set_ylabel('F1 score', fontsize=28)
#     plt.grid(axis="y", alpha=0.8, linestyle=':')
#
#     plt.savefig(str(name) + '.pdf')
#
#     # 对保存的图片进行显示
#     plt.show()
def motivation_3(path,name):
    #读取数据
    df = pd.read_excel(path)
    x = df['years'].values
    # y1 = df['V_precision'].values
    # y2 = df['V_recall'].values
    y3 = df['V_f1'].values
    # y4 = df['F_precision'].values
    # y5 = df['F_recall'].values
    y6 = df['F_f1'].values

    #设置画布大小以及x轴和y轴
    fig = plt.figure(figsize=(11, 10), dpi=80)
    ax = fig.add_axes([0.12, 0.24, 0.7, 0.6], zorder=11)
    dis=[1.5,1.8]
    ax.set_xticks(dis)
    ax.set_xticklabels(['2013-2015','2021-2023'], fontdict={'horizontalalignment': 'center', 'size': 28})
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_yticklabels([0.6,0.7,0.8,0.9,1],
                       fontdict={'horizontalalignment': 'right', 'size': 27})
    # plt.xlabel('Years', fontsize=28)

    ax.set_ylim((0.6,0.95))

    #画图
    # ax.plot(x, y3, linestyle=":", linewidth=5, label='Vulde', marker='X', markersize=20,
    #         color=seaborn.xkcd_rgb['dirty pink'], zorder=12)
    # ax.plot(dis, y6, linestyle="-", linewidth=5, label='FUNDED', marker='D', markersize=14,
    #         color=seaborn.xkcd_rgb['flat green'],zorder=12)
    # ax.bar(x=np.arange(len(x)), height=y1, label='TESSERACT', width=bar_width,
    #        linewidth=1, edgecolor='#8cc983', color='#a2d39b', zorder=10, alpha=0.9, hatch="||")
    dis2=[1.46,1.76]
    dis3 = [1.52, 1.82]
    dis4 = [1.58, 1.88]

    ax.bar([1.4,1.7], height=[0.932, 0.72112], label='Precision', width=0.05,
           linewidth=1, edgecolor='#3f7c37', color='#a2d39b', zorder=10, alpha=0.9, hatch="-")
    ax.bar(dis2, height=[0.8703, 0.6299395], label='Recall', width=0.05,
           linewidth=1, edgecolor='#3f7c37', color='#81c378', zorder=10, alpha=0.9, hatch="\\")
    ax.bar(dis3, height=[0.8454, 0.664825], label='F1 score', width=0.05,
           linewidth=1, edgecolor='#3f7c37', color='#60b454', zorder=10, alpha=0.9, hatch="//")
    ax.bar(dis4,height=[0.8552,0.694669528], label='Accuracy', width=0.05,
           linewidth=1, edgecolor='#3f7c37', color='#51a046', zorder=10, alpha=0.9, hatch="")
    # ax.bar(dis, height=[0.65, 0.87], label='Recall', width=0.05,
    #        linewidth=1, edgecolor='#8cc983', color='#a2d39b', zorder=10, alpha=0.9, hatch="//")
    # ax.bar(dis, height=[0.65, 0.87], label='F1', width=0.05,
    #        linewidth=1, edgecolor='#8cc983', color='#a2d39b', zorder=10, alpha=0.9, hatch="//")
    # 设置图例
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 28       }



    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # ax.legend(loc='center left', bbox_to_anchor=(0.22, 0.20), ncol=2, prop=font,columnspacing=0.8)
    ax.legend(loc='best', ncol=1, prop=font,columnspacing=1,handlelength=1.2)

    # ax.set_ylabel('F1 score', fontsize=28)
    plt.grid(axis="y", alpha=0.8, linestyle=':')

    plt.savefig(str(name) + '.pdf')

    # 对保存的图片进行显示
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an ellipse that represents the covariance matrix.
    'pos' is the mean position (center of the ellipse).
    'cov' is the covariance matrix.
    'nstd' is the number of standard deviations to determine the ellipse's radiuses.
    """
    if ax is None:
        ax = plt.gca()

    # Decompose and sort the eigenvalues and eigenvectors of covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The angle of rotation in degrees
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Width and height of the ellipse are scaled according to the eigenvalues
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_patch(ellipse)
    return ellipse
def motivation_2():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    # 设定随机种子以保证结果的可复现性
    # np.random.seed(0)

    # 14 漏洞
    mu1, sigma1 = [5, 5], [[1, 0.5], [0.5, 1]]  # 均值和协方差矩阵
    class1 = np.random.multivariate_normal(mu1, sigma1, 400)
    # Compute mean and covariance of the data
    data_mean_1 = np.mean(class1, axis=0)
    data_cov_1 = np.cov(class1, rowvar=False)
    # 14 非漏洞
    mu2, sigma2 = [7, 4], [[-0.6, 0.8], [0, 1]]
    class2 = np.random.multivariate_normal(mu2, sigma2, 400)
    data_mean_2 = np.mean(class2, axis=0)
    data_cov_2 = np.cov(class2, rowvar=False)
    # 24 非漏洞
    mu3, sigma3 = [7, 5], [[-0.1, 0.9], [0.2, 0.5]]
    class3 = np.random.multivariate_normal(mu3, sigma3, 400)
    data_mean_3 = np.mean(class3, axis=0)
    data_cov_3 = np.cov(class3, rowvar=False)
    # 24 漏洞
    mu4, sigma4 = [4, 6], [[-0.8, 0.4], [-0.2, 0.6]]
    class4 = np.random.multivariate_normal(mu4, sigma4, 400)
    data_mean_4 = np.mean(class4, axis=0)
    data_cov_4 = np.cov(class4, rowvar=False)
    # 绘图，设置图形尺寸为长方形
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(class4[:, 0], class4[:, 1], color='#4b9441', alpha=0.8, label='Vul. from 12-15', marker='o',s=200)
    plt.scatter(class3[:, 0], class3[:, 1], color='#0589d2', alpha=0.8, label='Benign from 15-18', marker='v',s=200)
    plt.scatter(class1[:, 0], class1[:, 1], color='#81c378', alpha=0.8, label='Vul. from 18-21', marker='o',s=200)
    plt.scatter(class2[:, 0], class2[:, 1], color='#74ccfc', alpha=0.8, label='Benign from 21-24', marker='v',s=200)

    # 添加背景颜色

    # plot_cov_ellipse(data_cov_1, data_mean_1, nstd=2.5, ax=ax, edgecolor='none', facecolor='#69b3ee', alpha=0.5)
    # plot_cov_ellipse(data_cov_2, data_mean_2, nstd=2.5, ax=ax, edgecolor='none', facecolor='#df615c', alpha=0.5)
    # plot_cov_ellipse(data_cov_3, data_mean_3, nstd=2, ax=ax, edgecolor='none', facecolor='#b22823', alpha=0.5)
    # plot_cov_ellipse(data_cov_4, data_mean_4, nstd=2, ax=ax, edgecolor='none', facecolor='#0e4c7d', alpha=0.5)

    # 添加图例和网格
    # font_size = 24
    # ax.legend(frameon=True, loc='top', fontsize=font_size,handlelength=0.1)
    ax.grid(True, linestyle='--', alpha=0.3)

    font = {'family': 'Arial',
            'weight': 'light',
            'size': 32
            }

    box = ax.get_position()
    # columnspacing=2
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='lower center', ncol=2, prop=font,handletextpad=0.1,
              )
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.25), ncol=2, prop=font, columnspacing=1, handletextpad=0.1,
              handlelength=1.15)
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(1)  # 设置线宽为2
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # 显示图形
    plt.savefig(r"E:\model_drift\fugures_plot\figure\motivation_distribution.pdf", bbox_inches='tight')
    plt.show()


motivation(r'E:\model_drift\fugures_plot\data\mock_data.xlsx',r'E:\model_drift\fugures_plot\figure\motivation')
# motivation_2()
# motivation_3(r'E:\model_drift\fugures_plot\data\mock_data.xlsx',r'E:\model_drift\fugures_plot\figure\motivation_2')