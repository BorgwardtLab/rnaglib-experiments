import json
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines
import matplotlib.patches as mpatches

plt.rcParams["text.usetex"] = True
plt.rc("font", size=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
plt.rc("grid", color="grey", alpha=0.2)

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
                        "threshold": f"{threshold:.1f}",
                        "seed": seed,
                        "distance": distance,
                    }
                )

# Add random performance over three seeds
for threshold in thresholds:
    for seed in SEEDS:
        path = f"results/thresholds_rna_site_redundant_rand_0.4_{seed}.json"
        with open(path) as result:
            result = json.load(result)
            score = result[METRIC]
            rows.append(
                {
                    "score": score,
                    "threshold": f"{threshold:.1f}",
                    "seed": seed,  # simulate different seeds
                    "distance": "rand",  # used for filtering
                }
            )

# Build DataFrame
df = pd.DataFrame(rows)
df.to_csv("thresholds_with_random.csv")

# Create a column to use for legend hue, exclude 'rand'
# df["distance"] = df["distance"].apply(lambda d: d if d in ["struc", "seq"] else "random")

# Custom color palette (you can tweak these)
palette = {
    "struc": sns.color_palette()[0],
    "seq": sns.color_palette()[1],
    "random": sns.color_palette()[2],
}

dist_names = {
    "struc": r"Structure",
    "seq": r"Sequence",
    "rand": r"Random"
}
df["distance"] = df["distance"].replace(dist_names)

palette_dict = sns.color_palette("muted")
palette_dict = {
    r"Structure": palette_dict[0],
    r"Sequence": palette_dict[9],
    r"Random": palette_dict[3],
}
print(df)

# df_no_rand = df[df["distance"] != r"Random"]
ax = sns.lineplot(
    data=df,
    x="threshold",  # X-axis is the threshold
    y="score",  # Y-axis is the score
    hue="distance",  # Differentiate lines by distance
    errorbar='sd',  # Use standard deviation for the error band
    legend=False,
    palette=palette_dict,
    linewidth=2,
)

# Set axis labels and title
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.tight_layout()

# # Format axes
# g.set_axis_labels("Threshold", "Test Score")
# g.set(ylim=(0.5, 0.85))
sns.despine()
#
# # Save + show
plt.savefig("plotting_scripts/thresholds.pdf", format="pdf")
plt.show()
plt.clf()
