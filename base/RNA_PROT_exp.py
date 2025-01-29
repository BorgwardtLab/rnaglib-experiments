"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ProteinBindingSite
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_RBP = ProteinBindingSite("RNA_Prot", recompute=False, debug=False)

ta_RBP.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Setup model

models_RBP = [
    PygModel(
        ta_RBP.metadata["description"]["num_node_features"],
        ta_RBP.metadata["description"]["num_classes"],
        graph_level=False,
        num_layers=i,
    )
    for i in range(3)
]
