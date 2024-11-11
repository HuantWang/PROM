import numpy as np
import matplotlib.pyplot as plt

def ae_line_gaussian():

    def f(x, c):
        return np.exp(-((x - 1) ** 2) / (2 * x * c ** 2))


    x_values = np.linspace(0.1, 5, 500)


    c_values = [0,1,2,3,4]


    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

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


    ax.set_xticks(np.arange(0, 5.1))
    ax.set_xticklabels([0, 1, 2, 3, 4, 5], fontdict={'horizontalalignment': 'center', 'size': 28})
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([0,0.2,0.4,0.6, 0.8, 1], fontdict={'horizontalalignment': 'right', 'size': 27})


    ax.set_xlabel('Prediction set size', fontsize=28)
    ax.set_ylabel('Confidence score', fontsize=28)


    ax.set_ylim((0, 1.1))


    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 28}
    ax.legend(loc='upper center', bbox_to_anchor=(0.41, 1.22), ncol=4,
              prop=font, columnspacing=0.2, handletextpad=0.03,
              handlelength=1.75)


    ax.grid(axis="y", alpha=0.8, linestyle=':')


    plt.tight_layout()


    plt.savefig(r'./figures_plot/figure/gaussian.pdf')
    plt.show()
    print("13 (a): Prom performance as Gaussian scale parameter changes.")
