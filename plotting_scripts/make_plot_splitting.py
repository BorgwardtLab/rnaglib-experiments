import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines
import matplotlib.patches as mpatches
import latex

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from constants import TASKLIST, SEEDS, METRICS, SPLITS

plt.rcParams["text.usetex"] = True
#plt.rcParams['font.family'] = 'monospace'
plt.rc("font", size=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
plt.rc("grid", color="grey", alpha=0.2)

rows = []
for task in TASKLIST:
    for distance in SPLITS:
        for seed in SEEDS:
            path = f"../../results/{task}_{distance}_2.5D_best_params_seed{seed}_results.json"

            with open(path) as result:
                result = json.load(result)
                metric_key = METRICS[task.split("_redundant")[0]]
                print(task)
                print(result["test_metrics"])
                score = (
                    result["test_metrics"][metric_key]
                )
                rows.append(
                    {
                        "score": score,
                        "metric": metric_key,
                        "task": task,
                        "seed": seed,
                        "distance": distance,
                    }
                )

df = pd.DataFrame(rows)
df.to_csv("splitting_publication_final_benchmark.csv")
df_mean = df.groupby(["task", "distance"])["score"].mean().reset_index()
df_std = df.groupby(["task", "distance"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task.split("_redundant")[0]] for row in df_mean.itertuples()]

print(df_mean)

Replace label for prettier x-axis
task_names = {
    "rna_go": r"\texttt{go}",
    "rna_if": r"\texttt{if}",
    "rna_cm": r"\texttt{cm}",
    "rna_prot": r"\texttt{prot}",
    "rna_site": r"\texttt{site}",
    "rna_ligand": r"\texttt{ligand}",
    "rna_cm_redundant": r"\texttt{cm} \newline \textit{redundant}",
    "rna_site_redundant": r"\texttt{site} \newline \textit{redundant}",
}
# task_names = {
#     "rna_go": "go",
#     "rna_if": "if",
#     "rna_cm": "cm",
#     "rna_prot": "prot",
#     "rna_site": "site",
#     "rna_ligand": "ligand",
#     "rna_cm_redundant": "cm redundant",
#     "rna_site_redundant": "site redundant",
# }
df["task"] = df["task"].replace(task_names)

dist_names = {
    "struc": r"Structure",
    "seq": r"Sequence",
    "rand": r"Random"
}
df["distance"] = df["distance"].replace(dist_names)

palette_dict = sns.color_palette("muted")
# palette_dict = sns.color_palette()
# palette_dict = {
#     r"Structure": "#2ba9ff",
#     r"Sequence": "#0a14db",
#     r"Random": "#FA4828",
# }
palette_dict = {
    r"Structure": palette_dict[0],
    r"Sequence": palette_dict[9],
    r"Random": palette_dict[3],
}

g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="distance",
    kind="bar",
    height=4,
    aspect=1.6,
    legend=False,
    palette=palette_dict,
    order=task_names.values(),
)
g.set_axis_labels(x_var="", y_var="Test Score")
g.set(ylim=(0, 1))
# g.despine(left=True)

# Add a vertical dotted line between rna_site and rna_site_redundant
# ax = g.ax
# line_pos = list(task_names.keys()).index("rna_site") + 0.5
# ax.axvline(x=line_pos, ymax=0.75, linestyle="--", color="dimgray", linewidth=2)

# Create handles and labels manually
handles = []
labels = []
for i, distance in enumerate(dist_names.values()):
    # Create a dummy rectangle for each distance, using the color from the plot
    color = palette_dict[distance]  # Get color for this distance
    # color = sns.color_palette()[i]  # Get color for this distance
    # handle = mpatches.Circle((0, 0), radius=0.5, color=color)
    # Using mlines.Line2D with marker 'o'
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, )
    # label=distance_name) # Use Line2D with marker 'o' and set markersize
    # handle = plt.Rectangle((0, 0), 1, 1, color=color)  # Create a rectangle with that color
    handles.append(handle)
    labels.append(distance)
plt.legend(handles, labels, loc="upper center", ncol=3, title=r"Splitting strategy:", handletextpad=-0.3)

plt.subplots_adjust(bottom=0.15)  # Adjust the values as needed
plt.savefig("splitting_ablation.pdf", format="pdf")
plt.show()
plt.clf()
