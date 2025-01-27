"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from exp import RNATrainer

# Setup task
ta = RNAGo(root="RNA_GO", recompute=True, debug=False)
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders(batch_size=1)

# Create model
model = PygModel(
    ta.metadata["description"]["num_node_features"],
    num_classes=len(ta.metadata["label_mapping"]),
    graph_level=True,
    multi_label=True,
)

# Create trainer and run
trainer = RNATrainer(ta, model, wandb_project="rna_go")
trainer.train()
