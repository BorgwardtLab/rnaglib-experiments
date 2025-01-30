"""experiment setup."""

import os
import sys
import shutil

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import InverseFolding
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

# Setup task
ta_IF_struc = InverseFolding(root="RNA_IF_struc", recompute=False, debug=False)
ta_IF_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists("RNA_IF_seq"):
    ta_IF_seq = InverseFolding(root="RNA_IF_seq", recompute=False, debug=False)
else:
    ta_IF_seq = InverseFolding(root="RNA_IF_seq", recompute=False, debug=False)
    distance = CDHitComputer()
    ta_IF_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_IF_seq.dataset = distance(task_IF_seq.dataset)

    ta_IF_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_IF_seq.set_loaders(recompute=True)

    source = "RNA_IF_seq"
    tmp = "RNA_IF_tmp"

    temp_dir = os.path.join(os.path.dirname(source), tmp)
    shutil.copytree(source, tmp)
    shutil.rmtree(source)
    os.rename(tmp, source)



# Create model
models_IF = [
    {
        "num_node_features": num_node_features=ta_IF_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_IF_seq.metadata["description"]["num_classes"] + 1,
        "graph_level": False,
        "num_layers": i,
    }
    for i in range(3)
]
