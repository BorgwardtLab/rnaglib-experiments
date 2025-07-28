import os
import time

import pandas as pd
import torch

from rnaglib.tasks import *
from rnaglib.learning.task_models import PygModel
from rnaglib.tasks.RNA_VS.model_pyg import VSPygModel
from rnaglib.transforms import GraphRepresentation

TASKS = [
    VirtualScreening,
    RNAGo,
    ChemicalModification,
    LigandIdentification,
    ProteinBindingSite,
    BindingSite,
    InverseFolding,

]

resdir = "timing_exp"
os.makedirs(resdir, exist_ok=True)
rows = []
for device in ("cpu", "cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        continue
    for task in TASKS:
        print()
        print(f'Timing for {task.name}')
        epochs = 1 if task.name == "rna_if" else 3
        batch_size = 8

        # Inverse folding is slower, so we should not do 3 whole epochs
        ta = task(root=os.path.join(resdir, task.name))
        ta.add_representation(GraphRepresentation(framework="pyg"))
        ta.get_split_loaders(batch_size=batch_size)
        model = PygModel.from_task(ta, device=device) if ta.name != "rna_vs" else VSPygModel.from_task(ta,
                                                                                                       device=device)
        model.configure_training(learning_rate=0.001)

        # Now time and monitor training
        tic = time.time()
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats(device)

        model.train_model(ta, epochs=epochs)

        if device != "cpu":
            # Get memory and convert to megabytes (MB) for readability
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            peak_memory_mb = 0

        time_training = time.time() - tic
        time_per_point = time_training / len(ta.train_dataset)
        print(f"Time to train {task.name} model on {device} over 3 epochs: {time_training:.3f} seconds, "
              f"time per point: {time_per_point:.4f} seconds,"
              f" memory used: {peak_memory_mb:.2f} MB")
        rows.append({"task_name": task.name,
                     "time": time_per_point,
                     "memory": peak_memory_mb
                     })
df = pd.DataFrame(rows)
print(df)
