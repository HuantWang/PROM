import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def Individual(path, name):
    df = pd.read_excel(path)
    x = df['value'].values
    # threshold = 0.2
    # replacement_value = 0.21

    y1 = df['RAISE'].values


    s1 = df['sRAISE'].values


    b1 = df['bRAISE'].values

    fig = plt.figure(figsize=(10, 12), dpi=80)
    ax = fig.add_axes([0.2, 0.3, 0.7, 0.5], zorder=11)
    plt.tight_layout()
    ax.set_xticks(np.arange(0, 5, 1) )
    ax.set_xticklabels(x, fontdict={'horizontalalignment': 'center', 'size': 28,'family': 'Arial'})
    ax.set_yticks(np.arange(0, 0.051, 0.01))
    ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],
                       fontdict={'horizontalalignment': 'right', 'size': 28,'family': 'Arial'})
    ax.set_ylim((0, 0.05))
    # ax.set_ylabel('Accuracy', fontsize=22,fontdict={'family': 'Arial'})

    bar_width = 0.4
    ax.bar(x=np.arange(len(x)), height=y1, label='RISE', width=bar_width,
           linewidth=1, edgecolor='#8cc983', color='#4b9441', zorder=10, alpha=0.9, hatch="||")


    # 最大最小值
    ax.hlines(y=s1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)


    ax.hlines(y=b1, xmin=np.arange(len(x)) - 0.03, xmax=np.arange(len(x)) + 0.03, color='k', alpha=0.6, linewidth=2,
              zorder=12)


    ax.vlines(x=np.arange(len(x)), ymin=s1, ymax=b1, color='k', alpha=0.6, linewidth=2, zorder=12)




    font = {'family':'Arial',
            'weight': 'light',
            'size': 28
            }

    y_label = ax.set_ylabel('Coverage deviations', fontsize=28)
    y_label.set_position((y_label.get_position()[0], y_label.get_position()[1]))
    font = {'family': 'Arial', 'weight': 'light', 'size': 20}
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # ax.legend(loc='center left', bbox_to_anchor=(-0.13, 1.16), ncol=4, prop=font, columnspacing=1, handletextpad=0.5,
    #           handlelength=2)
    plt.grid(axis="y", alpha=0.8, linestyle=':')
    plt.tight_layout()
    plt.savefig(str(name) + '.pdf')

    plt.show()


def ae_cd_plot_script(case=''):
    Individual(r'./figures_plot/data/ae_coverage.xlsx',r'./figures_plot/figure/coverage')
    print("Figure 13 (b): Coverage deviations across 5 case studies")