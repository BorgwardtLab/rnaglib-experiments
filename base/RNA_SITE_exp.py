"""experiment setup."""

import os
import sys
import shutil

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

# Setup task
ta_SITE_struc = BindingSite(root="RNA_SITE_struc", debug=False, recompute=False)
ta_SITE_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists("RNA_SITE_seq"):
    ta_SITE_seq = BindingSite(
        root="RNA_SITE_seq",
        recompute=False,
        debug=False,
    )
    ta_SITE_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_SITE_seq = BindingSite(
        root="RNA_SITE_seq",
        recompute=False,
        debug=False,
    )
    distance = CDHitComputer()
    ta_SITE_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_SITE_seq.dataset = distance(ta_SITE_seq.dataset)

    ta_SITE_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_SITE_seq.set_loaders(recompute=True)

    source = "RNA_SITE_seq"
    tmp = "RNA_SITE_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)


# Create model
models_SITE = [
    {
        "num_node_features": ta_SITE_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_SITE_seq.metadata["description"]["num_classes"],
        "graph_level": False,
        "num_layers": i,
    }
    for i in range(3)
]
