"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_ligand = LigandIdentification("RNA_Ligand", data_filename="binding_pockets.csv", recompute=False)
ta_ligand.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Create model
models_ligand = [
    PygModel(
        ta_ligand.metadata["description"]["num_node_features"],
        ta_ligand.metadata["description"]["num_classes"],
        graph_level=True,
        num_layers=i,
    )
    for i in range(3)
]
