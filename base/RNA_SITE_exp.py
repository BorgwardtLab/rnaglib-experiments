"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation

from exp import RNATrainer

# Setup task
ta = get_task(root="roots/RNA_Site", task_id="rna_site")

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders(batch_size=8)

model_args = {
    "num_node_features": ta.metadata["num_node_features"],
    "num_classes": ta.metadata["num_classes"],
    "graph_level": False,
    "num_layers": 4,
    "hidden_channels": 256
}

model = PygModel(**model_args)
trainer = RNATrainer(ta, model, epochs=100)

trainer.train()

if __name__ == "__main__":
    pass