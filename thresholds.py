import os
import json

from rnaglib.dataset_transforms.cd_hit import CDHitComputer
from rnaglib.dataset_transforms.structure_distance_computer import StructureDistanceComputer
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import ClusterSplitter, RandomSplitter
from rnaglib.learning import PygModel

from exp import RNATrainer


#TASKS_TODO = ['rna_site']


# Use this if you are submitting one job per task
TASKS_TODO = [os.environ.get('TASK')]
SIMILARITY_THRESHOLD = os.environ.get('SIMILARITY_THRESHOLD')

#TASKS_TODO = ['rna_site_redundant']
STRUCTURES_PATH = "/fs/pool/pool-wyss/RNA/.rnaglib/structures"

SPLITS = {"seq": 'cd_hit',
          "struc": 'USalign', 
          "rand": None,
          }

#SPLITS = {"rand": None}

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
                "rna_go": {"epochs": 20,
                           "learning_rate":0.0001}, #0.001 (original)
                "rna_if": {"epochs": 40, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
                           "learning_rate": 0.0001},
                "rna_ligand": {"epochs": 40,
                               "learning_rate": 1e-5},
                "rna_prot": {"epochs": 40, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
                            "learning_rate": 0.001}, #0.01 (original)
                "rna_site": {"batch_size": 8,
                             "epochs": 40}, # There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
                "rna_site_redundant": {"batch_size": 8,
                    "epochs": 40,
                         "learning_rate": 0.001}
                         }


recompute = True

for tid in TASKS_TODO:
    for split, distance in SPLITS.items():
        print(tid, split)
        root = f"roots/thresholds/{tid}_{split}_{SIMILARITY_THRESHOLD}"
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

        if distance not in task.dataset.distances:
            if split == 'struc':
                task.dataset = StructureDistanceComputer(structures_path=STRUCTURES_PATH)(task.dataset)
            if split == 'seq':
                task.dataset = CDHitComputer()(task.dataset)

        if split == 'rand':
            task.splitter = RandomSplitter()
        else:
            task.splitter = ClusterSplitter(distance_name=distance, similarity_threshold=float(SIMILARITY_THRESHOLD))
        # Representation needs to be added here as the loaders are not updated when the rep is added later.
        task.add_representation(GraphRepresentation(framework="pyg"))

        print('similarity threshold', SIMILARITY_THRESHOLD, type(SIMILARITY_THRESHOLD))
        task.get_split_loaders(recompute=True, batch_size=TRAINER_ARGS[tid]["batch_size"])

        task.write()




        for seed in [0,1,2]:
            model = PygModel.from_task(task, **MODEL_ARGS[tid])
            rep = GraphRepresentation(framework="pyg")
            result_file = f"results/thresholds_{tid}_{split}_{SIMILARITY_THRESHOLD}_{seed}.json"
            if os.path.exists(result_file) and not recompute:
                continue

            exp_name = f"{tid}_{split}_{SIMILARITY_THRESHOLD}_{seed}"

            trainer = RNATrainer(task, model, rep, seed=seed, wandb_project="rnaglib-thresholds", exp_name=exp_name, **TRAINER_ARGS[tid])
            trainer.train()
            metrics = model.evaluate(task, split="test")
            with open(result_file, "w") as j:
                json.dump(metrics, j)
                pass


