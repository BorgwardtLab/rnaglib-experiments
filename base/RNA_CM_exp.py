"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task, RNA_CM
from rnaglib.transforms import GraphRepresentation

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from exp import RNATrainer

# Setup task
ta = get_task(root="roots/RNA_CM", task_id="rna_cm")

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders(batch_size=8)

model_args = {
        "num_node_features": ta.metadata["num_node_features"],
        "num_classes": ta.metadata["num_classes"],
        "graph_level": False,
        "num_layers": 3,
}

model = PygModel(**model_args)
trainer = RNATrainer(ta, model, epochs=40)

trainer.train()

if __name__ == "__main__":
    pass
