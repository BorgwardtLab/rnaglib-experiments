import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines
import matplotlib.patches as mpatches
import latex

plt.rcParams["text.usetex"] = True
#plt.rcParams['font.family'] = 'monospace'
plt.rc("font", size=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
plt.rc("grid", color="grey", alpha=0.2)

TASKLIST = [
    "rna_cm", "rna_go", "rna_ligand", "rna_prot", "rna_if", "rna_site",
    "rna_site_redundant", "rna_cm_redundant"
]
DISTANCES = ["struc", "seq", "rand"]
SEEDS = [0, 1, 2]

METRICS = {
    "rna_cm": "balanced_accuracy",
    "rna_go": "jaccard",
    "rna_ligand": "balanced_accuracy",
    "rna_prot": "balanced_accuracy",
    "rna_if": "accuracy",
    "rna_site": "balanced_accuracy",
    "rna_site_redundant": "balanced_accuracy",
    "rna_cm_redundant": "balanced_accuracy"
}
NB_LAYER = {
    "rna_cm_redundant": {
        "struc": 3,
        "seq": 3,
        "rand": 6
    }
}
MODEL_ARGS = {
    "rna_cm_redundant": {
        "struc": {
            "num_layers": 3
        },
        "seq": {
            "num_layers": 3
        },
        "rand": {
            "num_layers": 6
        }
    },
    "rna_cm": {
        "struc": {
            "num_layers": 3,
            "hidden_channels": 128
        },
        "seq": {
            "num_layers": 3,
            "hidden_channels": 128
        },
        "rand": {
            "num_layers": 3,
            "hidden_channels": 128
        },
    },
    "rna_prot": {
        "struc": {
            "num_layers": 4,
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
        "seq": {
            "num_layers": 4,
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
        "rand": {
            "num_layers": 4,
            "hidden_channels": 64,
            "dropout_rate": 0.2
        },
    },
    "rna_site": {
        "struc": {
            "num_layers": 4,
            "hidden_channels": 256
        },
        "seq": {
            "num_layers": 4,
            "hidden_channels": 256
        },
        "rand": {
            "num_layers": 4,
            "hidden_channels": 256
        },
    },
    "rna_ligand": {
        "struc": {
            "num_layers": 3,
            "hidden_channels": 64,
        },
        "seq": {
            "num_layers": 3,
            "hidden_channels": 64,
        },
        "rand": {
            "num_layers": 3,
            "hidden_channels": 64,
        },
    }
}
TRAINER_ARGS = {
    "rna_cm": {
        "seq": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.001
        },
        "struc": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.001
        },
        "rand": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.001
        },
    },
    "rna_prot": {
        "rand": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.01
        },
        "seq": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.01
        },
        "struc": {
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.01
        },
    },  # 0.01 (original)
    "rna_site": {
        "rand": {
            "batch_size": 8,
            "epochs": 40,
            "learning_rate": 0.001
        },
        "seq": {
            "batch_size": 8,
            "epochs": 40,
            "learning_rate": 0.001
        },
        "struc": {
            "batch_size": 8,
            "epochs": 40,
            "learning_rate": 0.001
        },
    },
    "rna_ligand": {
        "rand": {
            "batch_size": 8,
            "epochs": 20,
            "learning_rate": 0.001
        },
        "seq": {
            "batch_size": 8,
            "epochs": 20,
            "learning_rate": 0.001
        },
        "struc": {
            "batch_size": 8,
            "epochs": 20,
            "learning_rate": 0.001
        },
    }
}
rows = []
for task in TASKLIST:
    for distance in DISTANCES:
        for seed in SEEDS:
            # buggy threshold
            if task == "rna_go":
                path = f"results/RNA_GO_{distance}_threshold0.6_3layers_lr0.001_20epochs_seed{seed}_results.json"
            elif task == "rna_cm_redundant":
                path = f"""results/RNA_CM_{MODEL_ARGS[task][distance]["num_layers"]}layers_lr0.001_40epochs_hiddendim128_2.5D_layer_type_rgcn_redundant_{distance}_seed{seed}_results.json"""
            elif task == "rna_cm":
                path = f"""results/RNA_CM_{MODEL_ARGS[task][distance]["num_layers"]}layers_lr{TRAINER_ARGS[task][distance]["learning_rate"]}_{TRAINER_ARGS[task][distance]["epochs"]}epochs_hiddendim{MODEL_ARGS[task][distance]["hidden_channels"]}_2.5D_layer_type_rgcn_{distance}_seed{seed}_results.json"""
            elif task == "rna_ligand":
                path = f"""results/rna_ligand_{MODEL_ARGS[task][distance]["num_layers"]}layers_lr{TRAINER_ARGS[task][distance]["learning_rate"]}_{TRAINER_ARGS[task][distance]["epochs"]}epochs_hiddendim{MODEL_ARGS[task][distance]["hidden_channels"]}_2.5D_layer_type_rgcn_batchsize8_{distance}_seed{seed}_results.json"""
            else:
                path = f"results/outerseed_{task}_{distance}_{seed}.json"

            with open(path) as result:
                result = json.load(result)
                metric_key = METRICS[task]
                score = (
                    result["test_metrics"][metric_key]
                    if task == "rna_go" or task == "rna_cm_redundant" or task == "rna_cm" or task == "rna_ligand"
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
df.to_csv("plotting_scripts/splitting_publication_BA_updated_RNA_CM.csv")
df_mean = df.groupby(["task", "distance"])["score"].mean().reset_index()
df_std = df.groupby(["task", "distance"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task] for row in df_mean.itertuples()]

print(df_mean)

# Replace label for prettier x-axis
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
plt.savefig("plotting_scripts/splitting_publication_BA_updated_RNA_CM.pdf", format="pdf")
plt.show()
plt.clf()
