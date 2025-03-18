"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer
from exp import RNATrainer


# Setup task
ta = get_task(root="roots/RNA_GO", task_id="rna_go")
# Create model
model_args = {
        "num_node_features": ta.metadata["num_node_features"],
        "num_classes": len(ta.metadata["label_mapping"]),
        "graph_level": True,
        "multi_label": True,
        "num_layers": 3,
    }



ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders()

model = PygModel(**model_args)
trainer = RNATrainer(ta, model)

trainer.train()

if __name__ == "__main__":
    pass
