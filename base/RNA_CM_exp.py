"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from exp import RNATrainer

# Setup task
ta_CM = ChemicalModification(
    root="RNA_CM",
    recompute=True,
    debug=False,
)
ta_CM.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Setup model

model_CM = PygModel(
    ta.metadata["description"]["num_node_features"],
    ta.metadata["description"]["num_classes"],
    graph_level=False,
)
