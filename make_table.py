import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TASKLIST = ["rna_cm", "rna_go", "rna_ligand", "rna_prot", "rna_site", "rna_if"]
#RNAFM = [True, False]
RNAFM = [True]
#DISTANCES = ["USalign", "cd_hit"]
DISTANCES = ["struc", "seq"]
SEEDS = [0, 1, 2]
#LAYERS = [0, 1, 2]
LAYERS = [2]

METRICS = {
    "rna_cm": "balanced_accuracy",
    "rna_go": "jaccard",
    "rna_ligand": "auc",
    "rna_prot": "balanced_accuracy",
    "rna_site": "balanced_accuracy",
    "rna_if": "accuracy",
}

rows = []
for task in TASKLIST:
    for distance in DISTANCES:
        for rnafm in RNAFM:
            for seed in SEEDS:
                    with open(
                        f"results/workshop_{task}_{distance}_{seed}.json"
                    ) as result:
                        result = json.load(result)
                        print(task)
                        print(result)
                        rows.append(
                            {
                                "score": result[METRICS[task]],
                                "metric": METRICS[task],
                                "task": task,
                                "seed": seed,
                                "distance": distance,
                            }
                        )
    pass

df = pd.DataFrame(rows)
df.to_csv("workshop_results.csv")
df_mean = df.groupby(["task", "distance"])["score"].mean().reset_index()
#df_mean = df.groupby(["task"])["score"].mean().reset_index()
df_std = df.groupby(["task", "distance"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

print(df_mean)

g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="distance",
    kind="bar",
    height=4,
    aspect=0.6,
)
g.set_axis_labels("", "Test Score")
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0, 1))
g.despine(left=True)
plt.savefig("workshop_results_preprint.pdf", format="pdf")
plt.show()
plt.clf()
