"""experiment setup."""

import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

# Setup task
ta_CM_struc = ChemicalModification(
    root="RNA_CM_struc",
    recompute=False,
    debug=False,
)
ta_CM_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists("RNA_CM_seq"):
    ta_CM_seq = ChemicalModification(
        root="RNA_CM_seq",
        recompute=False,
        debug=False,
    )
else:
    ta_CM_seq = ChemicalModification(
        root="RNA_CM_seq",
        recompute=False,
        debug=False,
    )
    distance = CDHitComputer()
    ta_CM_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_CM_seq.dataset = distance(ta_CM_seq.dataset)

    ta_CM_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_CM_seq.set_loaders(recompute=True)

    task_GO_seq.write()

    source = "RNA_CM_seq"
    tmp = "RNA_CM_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)


# Setup model

models_CM = [
    {
        "num_node_features": ta_CM_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_CM_seq.metadata["description"]["num_classes"],
        "graph_level": False,
        "num_layers": i,
    }
    for i in range(3)
]
