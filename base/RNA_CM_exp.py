"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_CM = ChemicalModification(
    root="RNA_CM",
    recompute=False,
    debug=False,
)
ta_CM.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Setup model

models_CM = [
    PygModel(
        ta_CM.metadata["description"]["num_node_features"],
        ta_CM.metadata["description"]["num_classes"],
        graph_level=False,
        num_layers=i,
    )
    for i in range(3)
]
