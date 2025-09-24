import os
import sys

import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines
import pandas as pd
from pathlib import Path
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from constants import TASKLIST, SEEDS, METRICS, SPLITS, DEFAULT_SPLIT

os.makedirs('plots', exist_ok=True)

plt.rcParams["text.usetex"] = True
# plt.rcParams['font.family'] = 'monospace'
plt.rc("font", size=20)  # fontsize of the tick labels
plt.rc("ytick", labelsize=18)  # fontsize of the tick labels
plt.rc("xtick", labelsize=18)  # fontsize of the tick labels
plt.rc("grid", color="grey", alpha=0.2)

# Load results
rows = []
for task in TASKLIST:
    task_name = task.split("_redundant")[0]
    for distance in SPLITS:
        for seed in SEEDS:
            path = f"results/{task}_{distance}_2.5D_best_params_seed{seed}_results.json"
            with open(path) as result:
                result = json.load(result)
                metric_key = METRICS[task_name]
                score = (result["test_metrics"][metric_key])
                rows.append({"score": score,
                             "metric": metric_key,
                             "task": task,
                             "seed": seed,
                             "distance": distance, })

# Compute means
df = pd.DataFrame(rows)
df.to_csv("plots/splitting_ablation.csv")
df_mean = df.groupby(["task", "distance"])["score"].mean().reset_index()
df_std = df.groupby(["task", "distance"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task.split("_redundant")[0]] for row in df_mean.itertuples()]

# After df_mean and df_std are created, load dummy scores
dummy_scores = []
for task in TASKLIST:
    task_name = task.split("_redundant")[0]
    for distance in SPLITS:
        dummy_path = f"results/dummy_{task.split("_redundant")[0]}_struc.json"
        try:
            with open(dummy_path) as f:
                dummy_result = json.load(f)
                score = dummy_result[METRICS[task.split("_redundant")[0]]]
                dummy_scores.append({
                    "task": task,
                    "distance": distance,
                    "dummy_score": score
                })
        except FileNotFoundError:
            print(f"Warning: Dummy file not found for {task} with {distance}")
            dummy_scores.append({
                "task": task,
                "distance": distance,
                "dummy_score": None
            })

# Convert to DataFrame and merge with df_mean
df_dummy = pd.DataFrame(dummy_scores)
df_mean = pd.merge(df_mean, df_dummy, on=["task", "distance"])

# Filter the DataFrame to keep only the rows to report
valid_pairs = list(DEFAULT_SPLIT.items())
df_mean = df_mean[df_mean[['task', 'distance']].apply(tuple, axis=1).isin(valid_pairs)]

# Just reorder
df_mean['task'] = pd.Categorical(df_mean['task'], categories=DEFAULT_SPLIT.keys(), ordered=True)
df_mean = df_mean.sort_values('task').reset_index(drop=True)
print(df_mean)

# Now let's move on to plotting.

# Replace label for prettier x - axis
task_names = {
    "rna_go": r"\texttt{go}",
    "rna_if": r"\texttt{if}",
    "rna_cm": r"\texttt{cm}",
    "rna_prot": r"\texttt{prot}",
    "rna_site": r"\texttt{site}",
    "rna_ligand": r"\texttt{ligand}",
    "rna_cm_redundant": r"\texttt{cm} \newline \textit{redundant}",
    "rna_prot_redundant": r"\texttt{prot} \newline \textit{redundant}",
    "rna_site_redundant": r"\texttt{site} \newline \textit{redundant}",
}
df["task"] = df["task"].replace(task_names)

dist_names = {
    "struc": r"Structure",
    "seq": r"Sequence",
    "rand": r"Random"
}
df["distance"] = df["distance"].replace(dist_names)

palette_dict = sns.color_palette("muted")
# palette_dict = sns.color_palette()
# palette_dict = {r"Structure": "#2ba9ff", r"Sequence": "#0a14db", r"Random": "#FA4828"}

palette_dict_reds = sns.color_palette("Reds")
palette_dict_blue = sns.color_palette("Blues")
palette_dict_greens = sns.color_palette("Greens")

palette_dict = {
    r"Structure": palette_dict[0],
    r"Sequence": palette_dict[9],
    r"Random": palette_dict_reds[3],
}

g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="distance",
    kind="bar",
    height=4,
    aspect=4,
    legend=False,
    palette=palette_dict,
    order=task_names.values(),
)
g.set_axis_labels(x_var="", y_var="Test Score")
g.set(ylim=(0, 1))
# g.despine(left=True)

# Add a vertical dotted line between rna_site and rna_site_redundant
ax = g.ax
line_pos = list(task_names.keys()).index("rna_ligand") + 0.5
ax.axvline(x=line_pos, ymax=0.75, linestyle="--", color="dimgray", linewidth=2)

# Create handles and labels manually
# handles = [mlines.Line2D([], [], color='white', marker='o', linestyle='None', markersize=0)]
# labels = [r"Splitting strategy: "]
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
# plt.legend(handles, labels, loc="upper center", ncol=len(handles), handletextpad=-0.3)
plt.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.1),
           title=r"Splitting strategy:", handletextpad=-0.3)

plt.subplots_adjust(bottom=0.15)  # Adjust the values as needed
plt.savefig("plots/splitting_ablation.pdf", format="pdf")
plt.show()
plt.clf()
