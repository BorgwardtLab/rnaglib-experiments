"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

SEQ_ROOT = "roots/RNA_CM_seq"
STRUC_ROOT = "roots/RNA_CM_struc"

# Setup task
ta_CM_struc = ChemicalModification(
    root=STRUC_ROOT,
    recompute=False,
    debug=False,
)
ta_CM_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists(SEQ_ROOT):
    ta_CM_seq = ChemicalModification(
        root=SEQ_ROOT,
        recompute=False,
        debug=False,
    )
    ta_CM_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_CM_seq = ChemicalModification(
        root=SEQ_ROOT,
        recompute=False,
        debug=False,
    )
    distance = CDHitComputer()
    ta_CM_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_CM_seq.dataset = distance(ta_CM_seq.dataset)

    ta_CM_seq.splitter = ClusterSplitter(distance_name=distance.name)

    ta_CM_seq.set_loaders(recompute=True)

    ta_CM_seq.write()

    source = SEQ_ROOT
    tmp = "roots/RNA_CM_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)


model_CM = {
        "num_node_features": ta_CM_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_CM_seq.metadata["description"]["num_classes"],
        "graph_level": False,
        "num_layers": 3,
        }


if __name__ == "__main__":
    pass
