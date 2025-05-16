import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines
import matplotlib.patches as mpatches

plt.rcParams["text.usetex"] = True
plt.rc("font", size=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
plt.rc("grid", color="grey", alpha=0.2)

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
REPRESENTATIONS_debug = ["2D"] * 3

rows = []
for ta_name in TASKLIST:
    for i, representation in enumerate(REPRESENTATIONS):
        for seed in SEEDS:
            # TO REMOVE ONCE FILES ARE PRODUCED
            repr_temp = representation
            representation = REPRESENTATIONS_debug[i]

            json_name = (
                f"results/{ta_name}_{representation}_{MODEL_ARGS[ta_name][representation]['num_layers']}layers_"
                f"lr{TRAINER_ARGS[ta_name][representation]['learning_rate']}_{TRAINER_ARGS[ta_name][representation]['epochs']}epochs_"
                f"hiddendim{MODEL_ARGS[ta_name][representation]['hidden_channels']}_batch_size{TRAINER_ARGS[ta_name][representation]['batch_size']}_seed{seed}_results.json")

            representation = repr_temp
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

REPRESENTATIONS += ["2.5D+rnafm"]
for task in TASKLIST:
    for seed in SEEDS:
        with open(f"results/workshop_{task}_rnafm_{seed}.json") as result:
            result = json.load(result)
            print(task)
            print(result)
            rows.append(
                {
                    "score": result[METRICS[task]],
                    "metric": METRICS[task],
                    "task": task,
                    "seed": seed,
                    "representation": REPRESENTATIONS[-1],
                }
            )

df = pd.DataFrame(rows)
df.to_csv("plotting_scripts/repr.csv")
df_mean = df.groupby(["task", "representation"])["score"].mean().reset_index()
df_std = df.groupby(["task", "representation"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

task_names = {
    "rna_cm": r"\texttt{cm}",
    "rna_prot": r"\texttt{prot}",
    "rna_site": r"\texttt{site}",
}
df["task"] = df["task"].replace(task_names)
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
    legend=False
)
g.set_axis_labels("", "Test Score")
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0.5, None))
g.despine()

# Create handles and labels manually
handles = []
labels = []
for i, representation in enumerate(REPRESENTATIONS):
    # Create a dummy rectangle for each distance, using the color from the plot
    color = palette_dict[i]  # Get color for this distance
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, )
    # handle = plt.Rectangle((0, 0), 1, 1, color=color)  # Create a rectangle with that color
    handles.append(handle)
    labels.append(representation)
plt.legend(handles, labels, loc="upper center", ncol=5, title=r"Representation type :", handletextpad=-0.3)
plt.subplots_adjust(bottom=0.1)  # Adjust the values as needed

plt.savefig(f"plotting_scripts/representations_ablation.pdf", format="pdf")
plt.show()
plt.clf()
