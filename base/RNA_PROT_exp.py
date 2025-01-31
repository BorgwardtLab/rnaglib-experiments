"""experiment setup."""

import os
import sys
import shutil

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ProteinBindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

# Setup task
ta_RBP_struc = ProteinBindingSite("RNA_RBP_struc", recompute=False, debug=False)
ta_RBP_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists("RNA_RBP_seq"):
    ta_RBP_seq = ProteinBindingSite(
        root="RNA_RBP_seq",
        recompute=False,
        debug=False,
    )
    ta_RBP_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_RBP_seq = ProteinBindingSite(
        root="RNA_RBP_seq",
        recompute=False,
        debug=False,
    )
    distance = CDHitComputer()
    ta_RBP_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_RBP_seq.dataset = distance(ta_RBP_seq.dataset)

    ta_RBP_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_RBP_seq.set_loaders(recompute=True)

    source = "RNA_RBP_seq"
    tmp = "RNA_RBP_seq"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)


# Setup model

models_RBP = [
    {
        "num_node_features": ta_RBP_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_RBP_seq.metadata["description"]["num_classes"],
        "graph_level": False,
        "num_layers": i,
    }
    for i in range(3)
]
