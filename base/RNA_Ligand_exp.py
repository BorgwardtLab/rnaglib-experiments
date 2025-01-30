"""experiment setup."""

import os
import sys
import shutil

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

# Setup task
ta_ligand_seq = LigandIdentification("RNA_Ligand_seq", data_filename="binding_pockets.csv", recompute=False)
ta_ligand_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists("RNA_Ligand_struc"):
    ta_ligand_struc = LigandIdentification("RNA_Ligand_struc", data_filename="binding_pockets.csv", recompute=False)
else:
    ta_ligand_struc = LigandIdentification("RNA_Ligand_struc", data_filename="binding_pockets.csv", recompute=False)
    distance = StructureDistanceComputer()
    ta_ligand_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_ligand_struc.dataset = distance(ta_ligand_seq.dataset)

    ta_ligand_struc.splitter = ClusterSplitter(distance_name=distance.name)

    ta_ligand_struc.set_loaders(recompute=True)

    source = "RNA_Ligand_struc"
    tmp = "RNA_Ligand_struc"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)


# Create model
models_ligand = [
    {
        "num_node_features": ta_ligand_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_ligand_seq.metadata["description"]["num_classes"],
        "graph_level": True,
        "num_layers": i,
    }
    for i in range(3)
]
