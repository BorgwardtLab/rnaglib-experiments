"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_GO = RNAGu(root="RNA_GO", recompute=True, debug=False)
ta_GO.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Create model
models_GO = [
    PygModel(
        ta.metadata["description"]["num_node_features"],
        num_classes=len(ta.metadata["label_mapping"]),
        graph_level=True,
        multi_label=True,
        num_layers=i,
    )
    for i in range(3)
]
