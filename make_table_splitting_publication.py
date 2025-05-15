import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TASKLIST = [
    "rna_cm", "rna_go", "rna_ligand", "rna_prot", "rna_if", "rna_site", "rna_site_redundant"
]
DISTANCES = ["struc", "seq", "rand"]
SEEDS = [0, 1, 2]

METRICS = {
    "rna_cm": "balanced_accuracy",
    "rna_go": "jaccard",
    "rna_ligand": "auc",
    "rna_prot": "balanced_accuracy",
    "rna_if": "accuracy",
    "rna_site": "balanced_accuracy",
    "rna_site_redundant": "balanced_accuracy",
}

rows = []

for task in TASKLIST:
    for distance in DISTANCES:
        for seed in SEEDS:
            if task == "rna_go":
                path = f"results/RNA_GO_{distance}_threshold0.6_3layers_lr0.001_20epochs_seed{seed}_results.json"
            else:
                path = f"results/workshop_{task}_{distance}_{seed}.json"

            with open(path) as result:
                result = json.load(result)
                metric_key = METRICS[task]
                score = (
                    result["test_metrics"][metric_key]
                    if task == "rna_go"
                    else result[metric_key]
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
df.to_csv("splitting_publication.csv")
df_mean = df.groupby(["task", "distance"])["score"].mean().reset_index()
df_std = df.groupby(["task", "distance"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

print(df_mean)


# Replace label for prettier x-axis
df["task"] = df["task"].replace({
    "rna_site_redundant": r"$\it{rna\_site}$" + "\n" + r"$\it{redundant}$"
})

# Set the task order manually to control where the vertical line goes
task_order = [
    "rna_cm", "rna_go", "rna_ligand", "rna_prot", "rna_if", "rna_site", 
    r"$\it{rna\_site}$\n$\it{redundant}$"
]
g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="distance",
    kind="bar",
    height=4,
    aspect=1.6,
)
g.set_axis_labels("Task", "Test Score")
g.set(ylim=(0, 1))
g.despine(left=True)

# Add a vertical dotted line between rna_site and rna_site_redundant
ax = g.ax
line_pos = task_order.index("rna_site") + 0.5
ax.axvline(x=line_pos, linestyle=":", color="black")

plt.savefig("splitting_publication.pdf", format="pdf")
plt.show()
plt.clf()