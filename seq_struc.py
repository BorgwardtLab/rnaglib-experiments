import os
import shutil

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


SPLITS = {"seq": 'cd_hit',
          "struc": 'USalign' 
          }

MODEL_ARGS = {"rna_cm": {"num_layers": 3},
              "rna_go": {"num_layers": 3},
              "rna_if": {"num_layers": 3},
              "rna_ligand": {"num_layers": 4},
              "rna_prot": {"num_layers": 3},
              "rna_site": {"num_layers": 4, 
                           "hidden_channels": 256},
              }

TRAINER_ARGS = {"rna_cm": {'epochs': 40, 
                           "batch_size": 8},
                "rna_go": {"epochs": 20,
                           "learning_rate":0.001},
                "rna_if": {"epochs": 100},
                "rna_ligand": {"epochs": 10,
                               "learning_rate": 1e-5},
                "rna_prot": {"epochs": 100},
                "rna_site": {"batch_size": 8, "epochs": 100}
                }




for tid in TASKS_TODO:
    for split, distance in SPLITS.items():
        print(tid, split)
        root = f"roots/{tid}_{split}"
        if os.path.exists("roots/{task}_{split}"): 
            task = get_task(task_id=tid, root=root)
        else:
            task = get_task(task_id=tid, root=root)
            task.splitter = ClusterSplitter(distance_name=distance)

            task.set_loaders(recompute=True)

            task.write()


        task.dataset.add_representation(GraphRepresentation(framework="pyg"))
        task.set_loaders(recompute=False)

        model_args = {
                "num_node_features": task.metadata["num_node_features"],
                "num_classes": task.metadata["num_classes"],
                "graph_level": task.metadata["graph_level"],
                "num_layers": 3,
                }

        task.get_split_loaders()

        model = PygModel.from_task(task, **MODEL_ARGS[tid])

        trainer = RNATrainer(task, model, **TRAINER_ARGS[tid])
        trainer.train()
        metrcs = model.evaluate(task, split="test")
