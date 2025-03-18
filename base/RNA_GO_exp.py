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
ta = get_task(root="roots/RNA_GO", task_id="rna_go")

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders(batch_size=8)

model_args = {
        "num_node_features": ta.metadata["num_node_features"],
        "num_classes": ta.metadata["num_classes"],
        "graph_level": True,
        "multi_label": True,
        "num_layers": 3,
}

model = PygModel(**model_args)
trainer = RNATrainer(ta, model, learning_rate=0.001, epochs=20)

trainer.train()

if __name__ == "__main__":
    pass