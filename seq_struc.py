import os
import json
import shutil

from rnaglib.dataset_transforms.cd_hit import CDHitComputer
from rnaglib.dataset_transforms.structure_distance_computer import StructureDistanceComputer
from rnaglib.tasks import get_task
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import ClusterSplitter
from rnaglib.learning import PygModel

from exp import RNATrainer


TASKS_TODO = ['rna_cm', 
              'rna_go',
              'rna_if',
              'rna_ligand',
              'rna_prot',
              'rna_site']

TASKS_TODO = ['rna_ligand']


SPLITS = {"seq": 'cd_hit',
          "struc": 'USalign' 
          }
MODEL_ARGS = {"rna_cm": {"num_layers": 3},
              "rna_go": {"num_layers": 3},
              "rna_if": {"num_layers": 3,
                         "hidden_channels": 128},
              "rna_ligand": {"num_layers": 4},
              "rna_prot": {"num_layers": 4, 
                          "hidden_channels": 64,
                          "dropout_rate": 0.2},
              "rna_site": {"num_layers": 4, 
                           "hidden_channels": 256},
              }

TRAINER_ARGS = {"rna_cm": {'epochs': 40, 
                           "batch_size": 8},
                "rna_go": {"epochs": 20,
                           "learning_rate":0.001},
                "rna_if": {"epochs": 100,
                           "learning_rate": 0.0001},
                "rna_ligand": {"epochs": 40,
                               "learning_rate": 1e-5},
                "rna_prot": {"epochs": 100, 
                            "learning_rate": 0.01},
                "rna_site": {"batch_size": 8,
                             "epochs": 100}
                         }


recompute = True

for tid in TASKS_TODO:
    for split, distance in SPLITS.items():
        print(tid, split)
        root = f"roots/{tid}_{split}"
        print(root)
        if os.path.exists(root): 
            task = get_task(task_id=tid, root=root)
        else:
            task = get_task(task_id=tid, root=root)

            if distance not in task.dataset.distances:
                if split == 'struc':
                    task.dataset = StructureDistanceComputer()(task.dataset)
                if split == 'seq':
                    task.dataset = CDHitComputer()(task.dataset)

            task.splitter = ClusterSplitter(distance_name=distance)
            task.get_split_loaders(recompute=True)

            task.write()


        model = PygModel.from_task(task, **MODEL_ARGS[tid])
        rep = GraphRepresentation(framework="pyg")

        for seed in [0, 1, 2]:
            result_file = f"results/workshop_{tid}_{split}_{seed}.json"
            if os.path.exists(result_file) and not recompute:
                continue
            trainer = RNATrainer(task, model, rep, seed=seed, **TRAINER_ARGS[tid])
            trainer.train()
            metrics = model.evaluate(task, split="test")
            with open(result_file, "w") as j:
                json.dump(metrics, j)
                pass


