import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_ARGS = {
    "rna_cm": {
        "hidden_channels": 128
    },
    "rna_prot": {
        "hidden_channels": 64,
        "dropout_rate": 0.2
    },
    "rna_site": {
        "hidden_channels": 256
    },
}

TRAINER_ARGS = {
    "rna_cm": {
        "epochs": 40, 
        "batch_size": 8,
        "learning_rate": 0.001
    },
    "rna_prot": {
        "epochs": 40, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
        "batch_size": 8,
        "learning_rate": 0.01
    }, #0.01 (original)
    "rna_site": {
        "batch_size": 8,
        "epochs": 40,
        "learning_rate":0.001
    }, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
}

METRICS = {
    "rna_cm": "balanced_accuracy",
    "rna_go": "jaccard",
    "rna_ligand": "auc",
    "rna_prot": "balanced_accuracy",
    "rna_site": "balanced_accuracy",
    "rna_if": "accuracy",
}
SEEDS = [0, 1, 2]
TASKLIST = ["rna_cm", "rna_site", "rna_prot"]
NB_LAYERS_LIST = [2,3,4,6]

rows = []
for ta_name in TASKLIST:
    for nb_layers in NB_LAYERS_LIST:
        for seed in SEEDS:
            json_name = f"""results/{ta_name}_seq_{str(nb_layers)}layers_lr{str(TRAINER_ARGS[ta_name]["learning_rate"])}_{str(TRAINER_ARGS[ta_name]["epochs"])}epochs_hiddendim{str(MODEL_ARGS[ta_name]["hidden_channels"])}_batch_size{str(TRAINER_ARGS[ta_name]["batch_size"])}_seed{str(seed)}_results.json"""
            with open(json_name) as result:
                result = json.load(result)
                test_metrics = result["test_metrics"]
                rows.append(
                    {
                        "score": test_metrics[METRICS[ta_name]],
                        "metric": METRICS[ta_name],
                        "task": ta_name,
                        "seed": seed,
                        "nb_layers": nb_layers,
                    }
                )
    pass

df = pd.DataFrame(rows)
df.to_csv("splitting.csv")
df_mean = df.groupby(["task", "nb_layers"])["score"].mean().reset_index()
df_std = df.groupby(["task", "nb_layers"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]
print(df_mean)
g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="nb_layers",
    kind="bar",
    height=4,
    aspect=1.6,
)
g.set_axis_labels("", "Test Score")
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0.5, None))
g.despine(left=True)

plt.savefig(f"nb_layers.pdf", format="pdf")
plt.show()
plt.clf()