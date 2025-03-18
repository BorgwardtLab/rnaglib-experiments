"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task 
from rnaglib.transforms import GraphRepresentation

from exp import RNATrainer

ta = get_task(root="roots/RNA_PROT", task_id="rna_prot")

model_args = {
        "num_node_features": ta.metadata["num_node_features"],
        "num_classes": ta.metadata["num_classes"],
        "graph_level": False,
        "num_layers": 3,
}


# Setup task
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders()

model = PygModel(**model_args)
trainer = RNATrainer(ta, model)

trainer.train()

if __name__ == "__main__":
    pass

