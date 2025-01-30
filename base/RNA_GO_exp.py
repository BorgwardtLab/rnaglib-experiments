"""experiment setup."""

import os
import sys
import shutil

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

# Setup task
ta_GO_seq = RNAGo(root="RNA_GO_seq", recompute=False, debug=False)
ta_GO_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists("RNA_GO_struc"):
    ta_GO_struc = RNAGo(
        root="RNA_GO_struc",
        recompute=False,
        debug=False,
    )
else:
    ta_GO_struc = RNAGo(
        root="RNA_GO_struc",
        recompute=False,
        debug=False,
    )
    distance = StructureDistanceComputer()
    ta_GO_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_GO_struc.dataset = distance(ta_GO_seq.dataset)

    ta_GO_struc.splitter = ClusterSplitter(distance_name=distance.name)

    ta_GO_struc.set_loaders(recompute=True)
    ta_GO_struc.write()

    source = "RNA_GO_struc"
    tmp = "RNA_GO_tmp"

    temp_dir = os.path.join(os.path.dirname(source), tmp)
    shutil.copytree(source, tmp)
    shutil.rmtree(source)
    os.rename(tmp, source)


# Create model
models_GO = [
    {
        "num_node_features": ta_GO_seq.metadata["description"]["num_node_features"],
        "num_classes": len(ta_GO_seq.metadata["label_mapping"]),
        "graph_level": True,
        "multi_label": True,
        "num_layers": i,
    }
    for i in range(3)
]
