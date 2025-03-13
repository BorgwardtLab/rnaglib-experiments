"""experiment setup."""

import os
import sys
import shutil

import torch
from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

SEQ_ROOT = "roots/RNA_SITE_seq"
STRUC_ROOT = "roots/RNA_SITE_struc"

# Setup task
ta_SITE_struc = BindingSite(root=STRUC_ROOT, debug=False, recompute=False)
ta_SITE_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists(SEQ_ROOT):
    ta_SITE_seq = BindingSite(
        root=SEQ_ROOT,
        recompute=False,
        debug=False,
    )
    ta_SITE_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_SITE_seq = BindingSite(
        root=SEQ_ROOT,
        recompute=False,
        debug=False,
    )
    distance = CDHitComputer()
    ta_SITE_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_SITE_seq.dataset = distance(ta_SITE_seq.dataset)

    ta_SITE_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_SITE_seq.set_loaders(recompute=True)

    source = SEQ_ROOT
    tmp = "roots/RNA_SITE_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)

# Create model
model_SITE = {
        "num_node_features": ta_SITE_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_SITE_seq.metadata["description"]["num_classes"],
        "graph_level": False,
        "num_layers": 3,
}
