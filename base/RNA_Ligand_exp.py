"""experiment setup."""

import os
import sys
import shutil

import torch
from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer

SEQ_ROOT = "roots/RNA_Ligand_seq"
STRUC_ROOT = "roots/RNA_Ligand_struc"

# Setup task
ta_ligand_seq = LigandIdentification(SEQ_ROOT, data_filename="binding_pockets.csv", recompute=False)
ta_ligand_seq.dataset.add_representation(GraphRepresentation(framework="pyg"))

if os.path.exists(STRUC_ROOT):
    ta_ligand_struc = LigandIdentification(STRUC_ROOT, data_filename="binding_pockets.csv", recompute=False)
    ta_ligand_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))
else:
    ta_ligand_struc = LigandIdentification(STRUC_ROOT, data_filename="binding_pockets.csv", recompute=False)
    distance = StructureDistanceComputer()
    ta_ligand_struc.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta_ligand_struc.dataset = distance(ta_ligand_seq.dataset)

    ta_ligand_struc.splitter = ClusterSplitter(distance_name=distance.name)

    ta_ligand_struc.set_loaders(recompute=True)

    source = STRUC_ROOT
    tmp = "roots/RNA_Ligand_tmp"

    shutil.copytree(source, tmp, dirs_exist_ok=True)
    shutil.rmtree(source)
    os.rename(tmp, source)

# Create model
model_ligand =  {
        "num_node_features": ta_ligand_seq.metadata["description"]["num_node_features"],
        "num_classes": ta_ligand_seq.metadata["description"]["num_classes"],
        "graph_level": True,
        "num_layers": 3,
    }
