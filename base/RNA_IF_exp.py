"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import InverseFolding
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

SEQ_ROOT = "roots/RNA_IF_seq"
STRUC_ROOT = "roots/RNA_IF_struc"

# Setup task
ta_IF_struc = InverseFolding(root=STRUC_ROOT, recompute=False, debug=False)
ta_IF_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists(SEQ_ROOT):
    ta_IF_seq = InverseFolding(root=SEQ_ROOT, recompute=False, debug=False)
    ta_IF_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_IF_seq = InverseFolding(root=SEQ_ROOT, recompute=False, debug=False)
    distance = CDHitComputer()
    ta_IF_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_IF_seq.dataset = distance(ta_IF_seq.dataset)

    ta_IF_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_IF_seq.set_loaders(recompute=True)

    source = SEQ_ROOT
    tmp = "roots/RNA_IF_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)


# Create model
model_IF =   {
        "num_node_features": ta_IF_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_IF_seq.metadata["description"]["num_classes"] + 1,
        "graph_level": False,
        "num_layers": 3,
    }
