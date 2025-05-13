import os
import json

from rnaglib.dataset_transforms.cd_hit import CDHitComputer
from rnaglib.dataset_transforms.structure_distance_computer import StructureDistanceComputer
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation, RNAFMTransform
from rnaglib.encoders import ListEncoder
from rnaglib.dataset_transforms import ClusterSplitter, RandomSplitter
from rnaglib.learning import PygModel

from rnaglib.tasks import RNAGo

from exp import RNATrainer


# TASKS_TODO = ['rna_cm', 
#               'rna_go',
#               'rna_if',
#               'rna_ligand',
#               'rna_prot',
#               'rna_site']


# Use this if you are submitting one job per task
TASKS_TODO = [os.environ.get('TASK')]

#TASKS_TODO = ['rna_site_redundant']

RNA_FM = [True, False]

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
              "rna_go_struc_0.6": {"num_layers": 3},
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
                "rna_prot": {"epochs": 100, 
                            "learning_rate": 0.001}, #0.01 (original)
                "rna_site": {"batch_size": 8,
                             "epochs": 100},
                "rna_go_struc_0.6": {"epochs": 100,
                             "learning_rate": 0.0001},   
                "rna_site_redundant": {"epochs": 100,
                               "learning_rate": 0.001}
                         }


recompute = True

for tid in TASKS_TODO:
    for rna_fm in RNA_FM:
        if rna_fm:
            root = f"roots/{tid}_rnafm"
        else:
            root = f"roots/{tid}_no_rnafm"
        print(root)

        if os.path.exists(root): 
            if tid != "rna_site_redundant":
                print(f"Loading task {tid} from {root}")
                task = get_task(task_id=tid, root=root)
            else:
                from rnaglib.tasks import BindingSiteRedundant
                print(f"Loading task {tid} from {root}")
                task = BindingSiteRedundant(root=root)
        else:
            if tid != "rna_site_redundant":
                print(f"Creating task {tid} in {root}")
                task = get_task(task_id=tid, root=root)
            else:
                from rnaglib.tasks import BindingSiteRedundant
                print(f"Creating task {tid} in {root}")
                task = BindingSiteRedundant(root=root)

        if rna_fm:
            rnafm = RNAFMTransform()
            [rnafm(rna) for rna in task.dataset]
            task.dataset.features_computer.add_feature(
                    feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)})
        # Representation needs to be added here as the loaders are not updated when the rep is added later.
        task.add_representation(GraphRepresentation(framework="pyg"))
        task.get_split_loaders(recompute=True)

        task.write()


        for seed in [0, 1, 2]:
            if rna_fm:
                model = PygModel.from_task(task, **MODEL_ARGS[tid], num_node_features=644)
            else:
                model = PygModel.from_task(task, **MODEL_ARGS[tid])
            rep = GraphRepresentation(framework="pyg")
            result_file = f"results/workshop_{tid}_{'rnafm' if rna_fm else 'no_rnafm'}_{seed}.json"
            if os.path.exists(result_file) and not recompute:
                continue

            exp_name = f"{tid}_{'rnafm' if rna_fm else 'no_rnafm'}_{seed}"

            trainer = RNATrainer(task, model, rep, seed=seed, wandb_project="rnaglib-embeddings", exp_name=exp_name, **TRAINER_ARGS[tid])
            trainer.train()
            metrics = model.evaluate(task, split="test")
            with open(result_file, "w") as j:
                json.dump(metrics, j)
                pass


