import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def set_box_colors(box, colors_box, edge_color_box):
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors_box[i])
        patch.set_edgecolor(edge_color_box[i])
    for i, whisker in enumerate(box['whiskers']):
        whisker.set_color(edge_color_box[i // 2])
    for i, cap in enumerate(box['caps']):
        cap.set_color(edge_color_box[i // 2])
    for median in box['medians']:
        median.set_color('black')
    for i, flier in enumerate(box['fliers']):
        flier.set(marker='o', color=edge_color_box[i // 2], alpha=0.5)


def plot_box_4_scores():
    color = ['red', 'green', 'blue', 'black']

    # Prepare data for plotting
    data_distri_1 = np.random.normal(0.9, 0.1, 10)
    data_less_than_1_1 = data_distri_1 < 1

    data_distri_2 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_2 = data_distri_2 < 1

    data_distri_3 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_3 = data_distri_3 < 1

    data_distri_4 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_4 = data_distri_4 < 1

    data_distri_5 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_5 = data_distri_5 < 1

    data_distri_6 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_6 = data_distri_6 < 1

    data_distri_7 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_7 = data_distri_7 < 1

    data_distri_8 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_8 = data_distri_8 < 1

    data_distri_9 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_9 = data_distri_9 < 1

    data_distri_10 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_10 = data_distri_10 < 1

    data_distri_11 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_11 = data_distri_11 < 1

    data_distri_12 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_12 = data_distri_12 < 1

    data_distri_13 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_13 = data_distri_13 < 1

    data_distri_14 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_14 = data_distri_14 < 1

    data_distri_15 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_15 = data_distri_15 < 1

    data_distri_16 = np.random.normal(0.8, 0.1, 500)
    data_less_than_1_16 = data_distri_16 < 1

    data_for_plotting = [data_distri_1, data_distri_2, data_distri_3, data_distri_4, data_distri_5, data_distri_6,
                         data_distri_7, data_distri_8]
    data_for_plotting_2 = [data_distri_9, data_distri_10, data_distri_11, data_distri_12, data_distri_13,
                           data_distri_14,
                           data_distri_15, data_distri_16]

    # Plot aesthetics settings
    figsize = (6, 3)
    box_widths = 0.3
    positions = [1, 2, 4, 5, 7, 8, 10, 11]
    duplicate_positions = [pos + 0.4 for pos in positions]

    colors_box = [color[0], color[1], color[0], color[1], color[0], color[1], color[0], color[1]]
    edge_color_box = [color[2], color[3], color[2], color[3], color[2], color[3], color[2], color[3]]
    # 为每种数据集创建一个图例标签
    legend_labels = ['Credibility score for right', 'Credibility score for wrong',
                     'Confidence score for right', 'Confidence score for wrong']
    legend_colors = [color[0], color[1], color[2], color[3]]  # 相应的颜色

    fig = plt.figure(figsize=figsize, dpi=80)
    ax = fig.add_axes([0.15, 0.2, 0.8, 0.6], zorder=11)

    # Create box plot
    box = plt.boxplot(data_for_plotting, positions=positions, widths=box_widths, patch_artist=True)
    box_dup = plt.boxplot(data_for_plotting_2, positions=duplicate_positions, widths=box_widths, patch_artist=True)

    # Customize colors and styles
    set_box_colors(box, colors_box, colors_box)
    set_box_colors(box_dup, edge_color_box, edge_color_box)

    font = {'family': 'Arial',
            'weight': 'light',
            'size': 16
            }

    # 创建代表每个数据集的矩形图形
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(legend_labels, legend_colors)]

    # 在图中添加图例
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, 1.5), loc='upper center', prop=font, ncol=2,
               columnspacing=0.5, handletextpad=0.1, handlelength=1.15)  # 你可以调整位置参数

    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.xticks([1.65, 4.65, 7.65, 10.65], ['SVM', 'MLP', 'LSTM', 'GNN'], fontsize=16)

    # Set y-axis limits
    plt.ylim(0, 1)
    plt.yticks(fontsize=16)
    # Add grid
    plt.grid(True)
    plt.show()


plot_box_4_scores()

