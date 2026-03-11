import matplotlib.pyplot as plt
import numpy as np

# =========================
# Global font settings
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Data
# Rows: AU, LD, HMEQ, TW, LC, GMSC
# Columns: FedAvg, FedProx, SCAFFOLD, FedNova, FedKD, FedCSL-CM
# =========================
data = np.array([
    [0.0450, 0.0860, 0.0560, 0.0667, 0.1820, 1.5100],
    [0.1166, 0.2392, 0.1166, 0.1166, 0.2600, 2.7943],
    [0.2825, 0.3188, 0.3166, 0.3166, 0.4236, 7.0852],
    [1.5656, 1.5592, 1.6166, 1.5666, 1.7037, 15.4293],
    [1.6119, 2.0619, 1.6880, 1.9220, 2.4213, 17.6500],
    [7.5762, 13.7641, 11.6166, 7.5666, 17.9719, 44.7485]
])

methods = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova', 'FedKD', 'FedCSL-CM']
titles = ['AU', 'LD', 'HMEQ', 'TW', 'LC', 'GMSC']

# =========================
# Figure settings
# =========================
fig, axes = plt.subplots(3, 2, figsize=(12, 14), facecolor='white')
axes = axes.flatten()

bar_color = '#cfe8dc'      # soft green
edge_color = 'black'

for i, ax in enumerate(axes):
    values = data[i]

    # Slight gap before the last bar
    x = np.arange(len(methods), dtype=float)
    x[-1] += 0.25

    bars = ax.bar(
        x, values,
        width=0.68,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.8
    )

    # Title
    ax.set_title(titles[i], fontsize=15, pad=10)

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=10)

    # Y ticks
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Grid
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # Limits
    ax.set_xlim(-0.6, x[-1] + 0.9)
    ax.set_ylim(0, max(values) * 1.18)

    # Y-axis label
    ax.set_ylabel('Value', fontsize=11)

    # Add value labels
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(values) * 0.02,
            f'{v:.4f}'.rstrip('0').rstrip('.'),
            ha='center',
            va='bottom',
            fontsize=9
        )

# Layout
plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)

# Save
plt.savefig('figure_4.png', dpi=600, bbox_inches='tight', facecolor='white')
# plt.savefig('figure_4.pdf', bbox_inches='tight', facecolor='white')
plt.show()