import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from constants import TASKLIST, SPLITS, SEEDS, METRICS

rows = []
for task in TASKLIST:
    for distance in SPLITS:
        for seed in SEEDS:
                with open(
                    f"../../results/{task}_{distance}_2.5D_best_params_seed{seed}_results.json"
                ) as result:
                    result = json.load(result)
                    metric_key = METRICS[task.split("_redundant")[0]]
                    score = (
                        result["test_metrics"][metric_key]
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
    pass

df = pd.DataFrame(rows)
df.to_csv("final_benchmark.csv")
df_mean = df.groupby(["task", "distance"])["score"].mean().reset_index()
#df_mean = df.groupby(["task"])["score"].mean().reset_index()
df_std = df.groupby(["task", "distance"])["score"].std().reset_index()
df_mean["std"] = df_std["score"]
df_mean["metric"] = [METRICS[row.task.split("_redundant")[0]] for row in df_mean.itertuples()]

print(df_mean)

# After df_mean and df_std are created

# Load dummy scores
dummy_scores = []
for task in TASKLIST:
    for distance in SPLITS:
        dummy_path = f"""../results/dummy_{task.split("_redundant")[0]}_struc.json"""
        try:
            with open(dummy_path) as f:
                dummy_result = json.load(f)
                score = dummy_result[METRICS[task.split("_redundant")[0]]]
                dummy_scores.append({
                    "task": task,
                    "distance": distance,
                    "dummy_score": score
                })
        except FileNotFoundError:
            print(f"Warning: Dummy file not found for {task} with {distance}")
            dummy_scores.append({
                "task": task,
                "distance": distance,
                "dummy_score": None
            })

# Convert to DataFrame and merge with df_mean
df_dummy = pd.DataFrame(dummy_scores)
df_mean = pd.merge(df_mean, df_dummy, on=["task", "distance"])

print(df_mean)

g = sns.catplot(
    data=df,
    x="task",
    y="score",
    hue="distance",
    kind="bar",
    height=4,
    aspect=1.6,
)
g.set_axis_labels("", "Test Score")
# g.set_titles("{col_name} {col_var}")
g.set(ylim=(0, 1))
g.despine(left=True)
plt.savefig("final_benchmark.pdf", format="pdf")
plt.show()
plt.clf()
