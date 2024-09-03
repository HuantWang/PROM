ax.bar(x=np.arange(len(x)), height=y1, label='LineVul', width=bar_width,
       edgecolor='#b08cee', linewidth=1, color='#cfb8f5', zorder=10,
       alpha=0.9, align="center")
ax.bar(x=np.arange(len(x)) + bar_width + 0.06, height=y2, label='Vulde', width=bar_width,
       edgecolor='#925fe8',
       linewidth=1, color='#b08cee', zorder=10, alpha=0.9, hatch='////')
ax.bar(x=np.arange(len(x)) + 2 * bar_width + 0.12, height=y3, label='Codexglue', width=bar_width,
       edgecolor='#7e41e4',
       linewidth=1, color='#9c6eea', zorder=10, alpha=0.9, hatch='|||')
ax.bar(x=np.arange(len(x)) + 3 * bar_width + 0.18, height=y4, label='FUNDED', width=bar_width,
       edgecolor='#6a23e0',
       linewidth=1, color='#8850e6', zorder=10, alpha=0.9, hatch='-')