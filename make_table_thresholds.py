import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config
thresholds = [0.4, 0.6, 0.8, 1]
SEEDS = [0, 1, 2]
DISTANCES = ["struc", "seq"]
METRIC = "balanced_accuracy"

rows = []

# Add threshold-based results for struc and seq
for threshold in thresholds:
    for distance in DISTANCES:
        for seed in SEEDS:
            path = f"results/thresholds_rna_site_redundant_{distance}_{threshold}_{seed}.json"
            with open(path) as result:
                result = json.load(result)
                score = result[METRIC]
                rows.append(
                    {
                        "score": score,
                        "metric": METRIC,
                        "task": f"{threshold:.1f}",
                        "seed": seed,
                        "distance": distance,
                    }
                )

# Add simulated "random" group using thresholds 0.4, 0.6, 0.8
rand_thresholds = [0.4, 0.6, 0.8]
for i, threshold in enumerate(rand_thresholds):
    path = f"results/thresholds_rna_site_redundant_rand_{threshold}_{0}.json"
    with open(path) as result:
        result = json.load(result)
        score = result[METRIC]
        rows.append(
            {
                "score": score,
                "metric": METRIC,
                "task": "random",
                "seed": i,  # simulate different seeds
                "distance": "rand",  # used for filtering
            }
        )

# Build DataFrame
df = pd.DataFrame(rows)
df.to_csv("thresholds_with_random.csv")

# Ensure x-axis order: random first, then thresholds
task_order = ["random"] + [f"{t:.1f}" for t in thresholds]

# Create a column to use for legend hue, exclude 'rand'
df["plot_distance"] = df["distance"].apply(lambda d: d if d in ["struc", "seq"] else "random")

# Custom color palette (you can tweak these)
palette = {
    "struc": sns.color_palette()[0],
    "seq": sns.color_palette()[1],
    "random": sns.color_palette()[2],
}

# Plot
g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="plot_distance",
    kind="bar",
    height=4,
    aspect=1,
    order=task_order,
    palette=palette,
)

# Format axes
g.set_axis_labels("Threshold", "Test Score")
g.set(ylim=(0.4, 0.85))
g.despine(left=True)

# Remove legend entirely
g._legend.remove()

# Save + show
plt.savefig("thresholds.pdf", format="pdf")
plt.show()
plt.clf()