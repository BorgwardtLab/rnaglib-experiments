"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ProteinBindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

STRUC_ROOT = "roots/RNA_RBP_struc"
SEQ_ROOT = "roots/RNA_RBP_seq"

# Setup task
ta_RBP_struc = ProteinBindingSite(STRUC_ROOT, recompute=False, debug=False)
ta_RBP_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists(SEQ_ROOT):
    ta_RBP_seq = ProteinBindingSite(
        root=SEQ_ROOT,
        recompute=False,
        debug=False,
    )
    ta_RBP_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_RBP_seq = ProteinBindingSite(
        root=SEQ_ROOT,
        recompute=False,
        debug=False,
    )
    distance = CDHitComputer()
    ta_RBP_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_RBP_seq.dataset = distance(ta_RBP_seq.dataset)

    ta_RBP_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_RBP_seq.set_loaders(recompute=True)

    source = SEQ_ROOT
    tmp = "roots/RNA_RBP_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)



model_RBP = {
        "num_node_features": ta_RBP_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_RBP_seq.metadata["description"]["num_classes"],
        "graph_level": False,
        "num_layers": 3,
}
