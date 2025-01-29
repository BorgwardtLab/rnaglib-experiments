"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import InverseFolding
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_IF = InverseFolding(root="RNA_IF", recompute=True, debug=False)
ta_IF.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Create model
models_IF = [
    PygModel(
        num_node_features=ta_IF.metadata["description"]["num_node_features"],
        num_classes=ta_IF.metadata["description"]["num_classes"] + 1,
        graph_level=False,
        num_layers=i,
    )
    for i in range(3)
]
