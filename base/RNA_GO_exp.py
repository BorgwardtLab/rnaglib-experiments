"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task
from rnaglib.transforms import GraphRepresentation, RNAFMTransform
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer, RandomSplitter
from rnaglib.encoders import ListEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir,".."))
from exp import RNATrainer

# Hyperparameters (to tune)
nb_layers = 3
learning_rate = 0.001
epochs = 20
split = "seq"
similarity_threshold = 0.6
rna_fm = False

# Experiment name
exp_name="RNA_GO_"+split+"_threshold"+str(similarity_threshold)+"_"+str(nb_layers)+"layers_lr"+str(learning_rate)+"_"+str(epochs)+"epochs"
if rna_fm:
        exp_name += "rna_fm"

# Setup task
ta_GO = get_task(root="roots/RNA_GO", task_id="rna_go")

if split=="struc":
        distance = "USalign"
else:
        distance = "cd_hit"

if distance not in ta_GO.dataset.distances:
        if split == 'struc':
                ta_GO.dataset = StructureDistanceComputer()(ta_GO.dataset)
        if split == 'seq':
                ta_GO.dataset = CDHitComputer()(ta_GO.dataset)
if split == 'rand':
        ta_GO.splitter = RandomSplitter()
else:
        ta_GO.splitter = ClusterSplitter(similarity_threshold=similarity_threshold, distance_name=distance)

if rna_fm:
        rnafm = RNAFMTransform()
        [rnafm(rna) for rna in ta_GO.dataset]
        ta_GO.dataset.features_computer.add_feature(feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)})
            
rep = GraphRepresentation(framework="pyg")
ta_GO.dataset.add_representation(rep)
ta_GO.get_split_loaders(batch_size=8, recompute=True)

model_GO_args = {
        "num_node_features": ta_GO.metadata["num_node_features"],
        "num_classes": ta_GO.metadata["num_classes"],
        "graph_level": True,
        "multi_label": True,
        "num_layers": nb_layers,
}
model_GO = PygModel(**model_GO_args)
trainer_GO = RNATrainer(ta_GO, model_GO, rep, exp_name=exp_name, learning_rate=learning_rate, epochs=epochs)

if __name__ == "__main__":
        for seed in [0,1,2]:
                model_GO = PygModel(**model_GO_args)
                trainer_GO = RNATrainer(ta_GO, model_GO, rep, exp_name=exp_name+"_seed"+str(seed), learning_rate=learning_rate, epochs=epochs, seed=seed)
                trainer_GO.train()
