"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task 
from rnaglib.transforms import GraphRepresentation

from exp import RNATrainer

ta = get_task(root="roots/RNA_IF", task_id="rna_if")
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders()

model_args  =   {
        "num_node_features": ta.metadata["num_node_features"],
        "num_classes": ta.metadata["num_classes"] + 1,
        "graph_level": False,
        "num_layers": 3,
    }
model = PygModel(**model_args)
trainer = RNATrainer(ta, model)

trainer.train()


# Create model

