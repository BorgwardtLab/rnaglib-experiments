"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.transforms import GraphRepresentation

# Setup task
ta_SITE = BindingSite(root="RNA_Site", debug=False, recompute=False)
ta_SITE.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Create model
models_SITE = [
    PygModel(
        ta_SITE.metadata["description"]["num_node_features"],
        ta_SITE.metadata["description"]["num_classes"],
        graph_level=False,
        num_layers=i,
    )
    for i in range(3)
]
