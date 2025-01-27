"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from exp import RNATrainer

# Setup task
ta = ChemicalModification(
    root="RNA_CM",
    recompute=True,
    debug=False,
)
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders(recompute=True)

# Setup model

model = PygModel(
    ta.metadata["description"]["num_node_features"],
    ta.metadata["description"]["num_classes"],
    graph_level=False,
)


# Create trainer and run
trainer = RNATrainer(ta, model, wandb_project="rna_cm")
trainer.train()
