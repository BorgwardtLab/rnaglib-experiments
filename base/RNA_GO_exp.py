"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from exp import RNATrainer

# Setup task
ta_GO = RNAGu(root="RNA_GO", recompute=True, debug=False)
ta_GO.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Create model
model_GO = PygModel(
    ta.metadata["description"]["num_node_features"],
    num_classes=len(ta.metadata["label_mapping"]),
    graph_level=True,
    multi_label=True,
)
