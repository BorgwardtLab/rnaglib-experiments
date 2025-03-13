import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TASKLIST = ["rna_cm", "rna_go", "rna_ligand", "rna_prot", "rna_site", "rna_if"]
#RNAFM = [True, False]
RNAFM = [True]
#DISTANCES = ["USalign", "cd_hit"]
DISTANCES = ["struc"]
SEEDS = [0, 1, 2]
#LAYERS = [0, 1, 2]
LAYERS = [2]

METRICS = {
    "rna_cm": "accuracy",
    "rna_go": "accuracy",
    "rna_ligand": "accuracy",
    "rna_prot": "accuracy",
    "rna_site": "accuracy",
    "rna_if": "accuracy",
}

rows = []
for task in TASKLIST:
    for distance in DISTANCES:
        for rnafm in RNAFM:
            for seed in SEEDS:
                for layer in LAYERS:
                    try:
                        with open(
                            f"results/preprint_100ep_{task}_rnafm-{rnafm}_distance-{distance}_layers-{layer}_seed-{seed}_results.json"
                        ) as result:
                            result = json.load(result)["test_metrics"]
                            print(result)
                            print(task)
                            rows.append(
                                {
                                    "score": result[METRICS[task]],
                                    "metric": METRICS[task],
                                    "task": task,
                                    "seed": seed,
                                    "gnn_layers": layer,
                                    "distance": distance,
                                    "rnafm": rnafm,
                                }
                            )
                    except FileNotFoundError:
                        continue
                    pass
    pass

df = pd.DataFrame(rows)
print(df)
#df_mean = df.groupby(["task", "gnn_layers", "distance", "rnafm"])["score"].mean().reset_index()
df_mean = df.groupby(["task"])["score"].mean().reset_index()
#df_std = df.groupby(["task", "gnn_layers", "distance", "rnafm"])["score"].std().reset_index()
df_std = df.groupby(["task"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

df_mean.to_csv("benchmark_results_preprint.csv")
print(df_mean)

g = sns.catplot(
    data=df_mean,
    x="task",
    y="score",
    kind="bar",
    height=4,
    aspect=0.6,
)
g.set_axis_labels("", "Test Score")
# g.set_xticklabels(["Men", "Women", "Children"])
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0, 1))
g.despine(left=True)
plt.savefig("benchmark_results_preprint.pdf", format="pdf")
plt.show()
plt.clf()

"""

for var in ["gnn_layers", "distance", "rnafm"]:
    g = sns.catplot(
        data=df_mean,
        x=var,
        y="score",
        col="task",
        hue=var,
        kind="bar",
        height=4,
        aspect=0.6,
    )
    g.set_axis_labels("", "Test Score")
    # g.set_xticklabels(["Men", "Women", "Children"])
    # g.set_titles("{col_name} {col_var}")
    g.set(ylim=(0, 1))
    g.despine(left=True)
    plt.savefig(var + ".pdf", format="pdf")
    plt.clf()

## LITERATURE

TASKLIST = ["rna_site_bench"]
RNAFM = [True, False]
SEEDS = [0, 1, 2]
LAYERS = [0, 1, 2]

METRICS = {
    "rna_site_bench": "accuracy",
}

rows = []
for task in TASKLIST:
    for rnafm in RNAFM:
        for seed in SEEDS:
            for layer in LAYERS:
                try:
                    with open(f"results/{task}_rnafm-{rnafm}_layers-{layer}_seed-{seed}_results.json") as result:
                        result = json.load(result)["test_metrics"]
                        rows.append(
                            {
                                "score": result[METRICS[task]],
                                "metric": METRICS[task],
                                "task": task,
                                "seed": seed,
                                "gnn_layers": layer,
                                "distance": distance,
                                "rnafm": rnafm,
                            }
                        )
                except FileNotFoundError:
                    continue
                pass
pass

df = pd.DataFrame(rows)
df_mean = df.groupby(["task", "gnn_layers", "distance", "rnafm"])["score"].mean().reset_index()
df_std = df.groupby(["task", "gnn_layers", "distance", "rnafm"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

df_mean.to_csv("benchmark_results.csv")
print(df_mean)

for var in ["gnn_layers", "rnafm"]:
    g = sns.catplot(
        data=df_mean,
        x=var,
        y="score",
        col="task",
        hue=var,
        kind="bar",
        height=4,
        aspect=0.6,
    )
    g.set_axis_labels("", "Test Score")
    # g.set_xticklabels(["Men", "Women", "Children"])
    # g.set_titles("{col_name} {col_var}")
    g.set(ylim=(0, 1))
    g.despine(left=True)
    plt.savefig(var + "_lit.pdf", format="pdf")
    plt.clf()
"""
