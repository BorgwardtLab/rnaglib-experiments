import os
import sys

import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines
import pandas as pd
from pathlib import Path
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from constants import REPRESENTATIONS, SEEDS, METRICS

os.makedirs('plots', exist_ok=True)

plt.rcParams["text.usetex"] = False
plt.rc("font", size=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
plt.rc("grid", color="grey", alpha=0.2)

rows = []
for ta_name in ["rna_cm", "rna_site", "rna_prot"]:
    for i, representation in enumerate(REPRESENTATIONS):
        for seed in SEEDS:
            json_name = (
                f"results/{ta_name}_struc_{representation}_best_params_seed{seed}_results.json")

            with open(json_name) as result:
                result = json.load(result)
                test_metrics = result["test_metrics"]
                rows.append(
                    {
                        "score": test_metrics[METRICS[ta_name]],
                        "metric": METRICS[ta_name],
                        "task": ta_name,
                        "seed": seed,
                        "representation": representation,
                    }
                )

df = pd.DataFrame(rows)
df.to_csv("plots/representation_ablation.csv")
df_mean = df.groupby(["task", "representation"])["score"].mean().reset_index()
df_std = df.groupby(["task", "representation"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task.split("_redundant")[0]] for row in df_mean.itertuples()]

task_names = {
    "rna_cm": r"\texttt{cm}",
    "rna_prot": r"\texttt{prot}",
    "rna_site": r"\texttt{site}",
}
df["task"] = df["task"].replace(task_names)

# task_reps = task_names.update({
task_reps = {
    "2D_GCN": r"2D",
    "2D": r"2D+",
    "GVP_2.5D": r"GVP-2.5D",
}
df["representation"] = df["representation"].replace(task_reps)
print(df)

palette_dict = sns.color_palette("Reds")

g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="representation",
    kind="bar",
    height=4,
    aspect=1.6,
    palette=palette_dict,
    legend=False,
    order=task_names.values()
)
g.set_axis_labels("", "Test Score")
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0.45, 0.75))
plt.axhline(0.5, color='dimgray', linestyle='--')
g.despine()

# Create handles and labels manually
handles = []
labels = []
for i, representation in enumerate(df["representation"].unique()):
    # Create a dummy rectangle for each distance, using the color from the plot
    color = palette_dict[i]  # Get color for this distance
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, )
    # handle = plt.Rectangle((0, 0), 1, 1, color=color)  # Create a rectangle with that color
    handles.append(handle)
    labels.append(representation)
plt.legend(handles, labels, loc="upper center", ncol=5, title=r"Representation type :", handletextpad=-0.3)
plt.subplots_adjust(bottom=0.1)  # Adjust the values as needed

plt.savefig(f"plots/representation_ablation.pdf", format="pdf")
plt.show()
plt.clf()
