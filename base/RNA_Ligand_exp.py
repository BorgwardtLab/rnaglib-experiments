"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from exp import RNATrainer

# Setup task
batch_size = 8
ta = LigandIdentification("RNA_Ligand", data_filename="binding_pockets.csv", recompute=True)
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.set_loaders(batch_size=batch_size, recompute=True)

# Create model
model = PygModel(
    ta.metadata["description"]["num_node_features"],
    ta.metadata["description"]["num_classes"],
    graph_level=True,
)


# Create trainer and run
trainer = RNATrainer(ta, model, wandb_project="rna_ligand")
trainer.train()
