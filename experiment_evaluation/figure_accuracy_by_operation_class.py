"""Figure: accuracy by operation class (mean of five models, question-clustered 95% bootstrap intervals).
Values are those printed by operation_taxonomy_analysis.py (operation_taxonomy_summary.txt).

Usage: python -m experiment_evaluation.figure_accuracy_by_operation_class
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# Mean accuracy over the five models with question-clustered 95% bootstrap intervals
# (experiment_results_revision_experiments/analysis/operation_taxonomy_summary.txt)
classes = ["Direct lookup\n(12)", "Ordered lookup\n(160)", "Event precedence\n(86)", "Aggregation\n(209)", "Comparison\n(141)", "Causal diagnosis\n(18)"]
data = {
 "BM25, 5 lines":      [(83.3,58.3,100.0),(41.9,34.5,49.0),(4.2,0.9,8.1),(13.2,9.3,17.3),(34.3,27.8,41.3),(12.2,0.0,27.8)],
 "BM25, 50 lines":     [(91.7,78.3,100.0),(45.4,38.9,51.9),(4.9,2.1,8.4),(21.1,16.9,25.6),(40.6,34.6,47.0),(22.2,8.9,38.9)],
 "Robust-tool agent":  [(38.3,23.3,53.3),(43.4,38.9,48.0),(7.4,4.4,10.9),(39.9,35.2,44.5),(46.2,41.6,50.8),(22.2,12.2,32.2)],
 "Code-writing agent": [(18.3,11.7,25.0),(28.6,24.2,33.1),(8.8,6.0,11.9),(24.1,20.4,27.7),(40.0,34.9,44.8),(11.1,5.6,17.8)],
}
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
hatches = ["", "////", "", "...."]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 7.5})
fig, ax = plt.subplots(figsize=(7.0, 2.5))
x = np.arange(len(classes)); w = 0.2
for i, (name, vals) in enumerate(data.items()):
    v = np.array(vals)
    pos = x + (i - 1.5) * w
    ax.bar(pos, v[:, 0], w * 0.9, color=colors[i], hatch=hatches[i], edgecolor="white", linewidth=0.4, label=name, zorder=2)
    ax.errorbar(pos, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="#444441", elinewidth=0.6, capsize=1.2, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=7.5)
ax.set_ylabel("Accuracy (%), mean of 5 models"); ax.set_ylim(0, 100)
ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.5, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#888780"); ax.spines["bottom"].set_color("#888780")
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper right", handlelength=1.4, columnspacing=0.8)
fig.tight_layout(pad=0.2)
fig.savefig("experiment_results_revision_experiments/analysis/accuracy_by_operation_class.pdf"); 
