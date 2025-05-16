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
            "hidden_channels": 128
        },
        "2D": {
            "hidden_channels": 128
        }
    },
    "rna_prot": {
        "2.5D": {
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
        "2D": {
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
    },
    "rna_site": {
        "2.5D": {
            "hidden_channels": 256
        },
        "2D": {
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
NB_LAYERS_LIST = [2, 3, 4, 5, 6]

representation = "2D"

rows = []
for ta_name in TASKLIST:
    for nb_layers in NB_LAYERS_LIST:
        for seed in SEEDS:
            json_name = (f"results/{ta_name}_{representation}_{nb_layers}layers_"
                         f"lr{TRAINER_ARGS[ta_name][representation]['learning_rate']}_"
                         f"{TRAINER_ARGS[ta_name][representation]['epochs']}epochs_"
                         f"hiddendim{MODEL_ARGS[ta_name][representation]['hidden_channels']}_"
                         f"batch_size{TRAINER_ARGS[ta_name][representation]['batch_size']}_"
                         f"seed{seed}_results.json")
            with open(json_name) as result:
                result = json.load(result)
                test_metrics = result["test_metrics"]
                rows.append(
                    {
                        "score": test_metrics[METRICS[ta_name]],
                        # "metric": METRICS[ta_name],
                        "task": ta_name,
                        "seed": seed,
                        "nb_layers": nb_layers,
                    }
                )
df = pd.DataFrame(rows)
df.to_csv(f"plotting_scripts/nb_layers_{representation}.csv")
df_mean = df.groupby(["task", "nb_layers"])["score"].mean().reset_index()
df_std = df.groupby(["task", "nb_layers"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
# df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

# Replace label for prettier x-axis
task_names = {
    "rna_cm": r"\texttt{cm}",
    "rna_prot": r"\texttt{prot}",
    "rna_site": r"\texttt{site}",
}
df["task"] = df["task"].replace(task_names)

print(df)

palette_dict = sns.color_palette("Blues")
g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="nb_layers",
    kind="bar",
    height=4,
    aspect=1.6,
    palette=palette_dict,
    legend=False
)
g.set_axis_labels("", "Test Score")
g.set(ylim=(0.5, 0.7))
g.despine()

# Create handles and labels manually
handles = []
labels = []
for i, distance in enumerate(NB_LAYERS_LIST):
    # Create a dummy rectangle for each distance, using the color from the plot
    color = palette_dict[i]  # Get color for this distance
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, )
    # handle = plt.Rectangle((0, 0), 1, 1, color=color)  # Create a rectangle with that color
    handles.append(handle)
    labels.append(distance)
plt.legend(handles, labels, loc="upper center", ncol=5, title=r"Number of layers :", handletextpad=-0.3)
plt.subplots_adjust(bottom=0.1)  # Adjust the values as needed

plt.savefig(f"plotting_scripts/nb_layers_{representation}.pdf", format="pdf")
plt.show()
plt.clf()
