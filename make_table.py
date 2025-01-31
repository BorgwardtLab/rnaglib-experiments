import json
import pandas as pd

TASKLIST = ["rna_cm", "rna_go", "rna_ligand", "rna_prot", "rna_site", "rna_if", "rna_site_bench"]
RNAFM = [True, False]
DISTANCES = ["USalign", "cd_hit"]
SEEDS = [0, 1, 2]
LAYERS = [0, 1, 2]

METRICS = {
    "rna_cm": "accuracy",
    "rna_go": "f1",
    "rna_ligand": "auroc",
    "rna_prot": "mcc",
    "rna_site": "auroc",
    "rna_if": "accuracy",
    "RNA_site_bench": "accuracy",
}

rows = []
for task in TASKLIST:
    for distance in DISTANCES:
        for rnafm in RNAFM:
            for seed in SEEDS:
                for layer in LAYERS:
                    try:
                        with open(
                            f"results/{task}_rnafm-{rnafm}_distance-{distance}_layers-{layer}_seed-{seed}_results.json"
                        ) as result:
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
