import os
import json

from rnaglib.dataset_transforms.cd_hit import CDHitComputer
from rnaglib.dataset_transforms.structure_distance_computer import StructureDistanceComputer
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import ClusterSplitter, RandomSplitter
from rnaglib.learning import PygModel

from exp import RNATrainer

import numpy as np


TASKS_TODO = ['rna_cm', 
              'rna_go',
              'rna_if',
              'rna_ligand',
              'rna_prot',
              'rna_site']


# Use this if you are submitting one job per task
#TASKS_TODO = [os.environ.get('TASK')]


SPLITS = {"struc": 'USalign'}

MODEL_ARGS = {"rna_cm": {"num_layers": 3},
              "rna_go": {"num_layers": 3,
                         "multi_label": True},
              "rna_if": {"num_layers": 3,
                         "hidden_channels": 128},
              "rna_ligand": {"num_layers": 4},
              "rna_prot": {"num_layers": 4, 
                          "hidden_channels": 64,
                          "dropout_rate": 0.2},
              "rna_site": {"num_layers": 4, 
                           "hidden_channels": 256},
              "rna_site_redundant": {"num_layers": 4, 
                           "hidden_channels": 256},
              }

TRAINER_ARGS = {"rna_cm": {'epochs': 40, 
                           "batch_size": 8},
                "rna_go": {"epochs": 10,
                           "learning_rate":0.001}, #0.001 (original)
                "rna_if": {"epochs": 40, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
                           "learning_rate": 0.0001},
                "rna_ligand": {"epochs": 40,
                               "learning_rate": 1e-5},
                "rna_prot": {"epochs": 40, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
                            "learning_rate": 0.001}, #0.01 (original)
                "rna_site": {"batch_size": 8,
                             "epochs": 40}, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
                "rna_site_redundant": {"epochs": 100,
                         "learning_rate": 0.001}
                         }


def evaluate_dummy(model, task, split="test"):
    dataloader = model.get_dataloader(task=task, split=split)
    mean_loss, all_preds, all_probs, all_labels = model.inference(loader=dataloader)  # get real labels and structure

    # Flatten for analysis
    if task.metadata["graph_level"]:
        flat_labels = np.stack(all_labels)
    else:
        flat_labels = np.concatenate(all_labels)

    if task.metadata["multi_label"]:
        # Mean label vector across samples
        majority_vector = (flat_labels.mean(axis=0) > 0.5).astype(np.float32)
        all_preds = [majority_vector] * len(all_labels)
        all_probs = [majority_vector] * len(all_labels)

    elif task.metadata["num_classes"] == 2:
        # Binary: Predict class 1 or 0 everywhere
        unique, counts = np.unique(flat_labels, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        p = majority_class * np.ones_like(flat_labels, dtype=np.float32)
        full_probs = np.stack([1 - p, p], axis=1)

        # Same split logic as in inference
        if not task.metadata["graph_level"]:
            ptr = np.cumsum([0] + [len(lbl) for lbl in all_labels])
            all_probs = [full_probs[ptr[i]:ptr[i+1]] for i in range(len(ptr) - 1)]
            all_preds = [np.full_like(lbl, majority_class) for lbl in all_labels]
        else:
            all_probs = full_probs
            all_preds = np.full_like(flat_labels, majority_class)

    else:
        # Multiclass
        num_classes = task.metadata["num_classes"]
        unique, counts = np.unique(flat_labels, return_counts=True)
        majority_class = unique[np.argmax(counts)]

        one_hot_probs = np.zeros((len(flat_labels), num_classes), dtype=np.float32)
        one_hot_probs[:, majority_class] = 1.0

        if not task.metadata["graph_level"]:
            ptr = np.cumsum([0] + [len(lbl) for lbl in all_labels])
            all_probs = [one_hot_probs[ptr[i]:ptr[i+1]] for i in range(len(ptr) - 1)]
            all_preds = [np.full_like(lbl, majority_class) for lbl in all_labels]
        else:
            all_probs = one_hot_probs
            all_preds = np.full_like(flat_labels, majority_class)

    # Metrics on dummy preds
    metrics = task.compute_metrics(all_preds, all_probs, all_labels)
    metrics["loss"] = 0  # Dummy model, no training
    return metrics


recompute = True

for tid in TASKS_TODO:
    for split, distance in SPLITS.items():
        print(tid, split)
        root = f"roots/{tid}_{split}"
        print(root)

        if os.path.exists(root): 
            if tid != "rna_site_redundant":
                print(f"Loading task {tid} from {root}")
                task = get_task(task_id=tid, root=root)
            else:
                from rnaglib.tasks import BindingSiteRedundant
                print(f"Loading task {tid} from {root}")
                task = BindingSiteRedundant(root=root, structures_path=STRUCTURES_PATH)
        else:
            if tid != "rna_site_redundant":
                print(f"Creating task {tid} in {root}")
                task = get_task(task_id=tid, root=root)
            else:
                from rnaglib.tasks import BindingSiteRedundant
                print(f"Creating task {tid} in {root}")
                task = BindingSiteRedundant(root=root, structures_path=STRUCTURES_PATH)

        task.add_representation(GraphRepresentation(framework="pyg"))
        task.get_split_loaders(recompute=False)


    
        model = PygModel.from_task(task, **MODEL_ARGS[tid])
        rep = GraphRepresentation(framework="pyg")
        result_file = f"results/dummy_{tid}_{split}.json"
        if os.path.exists(result_file) and not recompute:
            continue

        exp_name = f"dummy_{tid}_{split}"


        metrics = evaluate_dummy(model, task, split="test")
        with open(result_file, "w") as j:
            json.dump(metrics, j)
            pass
