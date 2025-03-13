"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer


SEQ_ROOT = "roots/RNA_GO_seq"
STRUC_ROOT = "roots/RNA_GO_struc"

# Setup task
ta_GO_seq = RNAGo(root=SEQ_ROOT, recompute=False, debug=False)
ta_GO_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists(STRUC_ROOT):
    ta_GO_struc = RNAGo(
        root=STRUC_ROOT,
        recompute=False,
        debug=False,
    )
    ta_GO_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_GO_struc = RNAGo(
        root=STRUC_ROOT,
        recompute=False,
        debug=False,
    )
    distance = StructureDistanceComputer()
    ta_GO_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_GO_struc.dataset = distance(ta_GO_seq.dataset)

    ta_GO_struc.splitter = ClusterSplitter(distance_name=distance.name)

    ta_GO_struc.set_loaders(recompute=True)
    ta_GO_struc.write()

    source = STRUC_ROOT
    tmp = "roots/RNA_GO_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)
# Create model
model_GO = {
        "num_node_features": ta_GO_seq.metadata["description"]["num_node_features"],
        "num_classes": len(ta_GO_seq.metadata["label_mapping"]),
        "graph_level": True,
        "multi_label": True,
        "num_layers": 3,
    }
