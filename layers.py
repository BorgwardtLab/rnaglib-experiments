"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from exp import RNATrainer

MODEL_ARGS = {"rna_cm": {"hidden_channels": 128},
              "rna_prot": {"hidden_channels": 64,
                           "dropout_rate": 0.2},
              "rna_site": {"hidden_channels": 256},
              }

# There are only marginal improvements running a hundred epochs, so we leave it at 40 for the splitting analysis
TRAINER_ARGS = {"rna_cm": {"epochs": 40,
                           "batch_size": 8,
                           "learning_rate": 0.001},
                "rna_prot": {"epochs": 40,
                             "batch_size": 8,
                             "learning_rate": 0.01},  # 0.01 (original)
                "rna_site": {"batch_size": 8,
                             "epochs": 40,
                             "learning_rate": 0.001},
                }

if __name__ == "__main__":
    for ta_name in ["rna_prot", "rna_cm", "rna_site"]:
        ta = get_task(root="roots/" + ta_name, task_id=ta_name)
        rep = GraphRepresentation(framework="pyg")
        ta.dataset.add_representation(rep)
        for seed in [0, 1, 2]:
            ta.get_split_loaders(batch_size=TRAINER_ARGS[ta_name]["batch_size"], recompute=True)
            for nb_layers in [2, 3, 4, 6]:
                exp_name = (f"{ta_name}_seq_{nb_layers}layers_lr{TRAINER_ARGS[ta_name]['learning_rate']}_"
                            f"{TRAINER_ARGS[ta_name]['epochs']}epochs_hiddendim{MODEL_ARGS[ta_name]['hidden_channels']}_"
                            f"batch_size{TRAINER_ARGS[ta_name]['batch_size']}")
                model = PygModel(num_node_features=ta.metadata["num_node_features"],
                                 num_classes=ta.metadata["num_classes"],
                                 graph_level=ta.metadata["graph_level"],
                                 num_layers=nb_layers, **MODEL_ARGS[ta_name])
                trainer = RNATrainer(ta, model, rep, exp_name=exp_name + "_seed" + str(seed),
                                     learning_rate=TRAINER_ARGS[ta_name]["learning_rate"],
                                     epochs=TRAINER_ARGS[ta_name]["epochs"], seed=seed,
                                     batch_size=TRAINER_ARGS[ta_name]["batch_size"])
                trainer.train()
