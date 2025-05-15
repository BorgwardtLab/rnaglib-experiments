import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_ARGS = {
    "rna_cm": {
        "2.5D": {
            "num_layers": 3,
            "hidden_channels": 128
        },
        "2D": {
            "num_layers": 3,
            "hidden_channels": 128
        }
    },
    "rna_prot": {
        "2.5D": {
            "num_layers": 4,
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
        "2D": {
            "num_layers": 4,
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
    },
    "rna_site": {
        "2.5D": {
            "num_layers": 4,
            "hidden_channels": 256
        },
        "2D": {
            "num_layers": 2,
            "hidden_channels": 128
        },
    },
}

# There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
TRAINER_ARGS = {
    "rna_cm": {
        "2.5D": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.001
        },
        "2D": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.001
        }
    },
    "rna_prot": {
        "2.5D": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.01
        },
        "2D": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.01
        },
    },  # 0.01 (original)
    "rna_site": {
        "2.5D": {
            "batch_size": 8,
            "epochs": 40,
            "learning_rate": 0.001
        },
        "2D": {
            "batch_size": 8,
            "epochs": 40,
            "learning_rate": 0.0001
        }
    },
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
REPRESENTATIONS = ["2D_GCN", "2D", "2.5D"]

rows = []
for ta_name in TASKLIST:
    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            json_name = f"""results/{ta_name}_{representation}_{str(TRAINER_ARGS[ta_name][representation]["num_layers"])}layers_lr{str(TRAINER_ARGS[ta_name][representation]["learning_rate"])}_{str(TRAINER_ARGS[ta_name][representation]["epochs"])}epochs_hiddendim{str(MODEL_ARGS[ta_name][representation]["hidden_channels"])}_batch_size{str(TRAINER_ARGS[ta_name][representation]["batch_size"])}_seed{str(seed)}_results.json"""
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
df.to_csv("splitting.csv")
df_mean = df.groupby(["task", "representation"])["score"].mean().reset_index()
df_std = df.groupby(["task", "representation"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]
print(df_mean)
g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="representation",
    kind="bar",
    height=4,
    aspect=1.6,
)
g.set_axis_labels("", "Test Score")
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0.5, None))
g.despine(left=True)

plt.savefig(f"representations_ablation.pdf", format="pdf")
plt.show()
plt.clf()
