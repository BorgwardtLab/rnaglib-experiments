"""experiment setup."""

import os
import sys
import shutil

import torch
from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation, RNAFMTransform
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer, RandomSplitter
from rnaglib.encoders import ListEncoder
from rnaglib.tasks.RNA_Ligand.prepare_dataset import PrepareDataset


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir,".."))
from exp import RNATrainer

# Hyperparameters (to tune)
nb_layers = 3
learning_rate = 1e-4
epochs = 100
split = "rand"
rna_fm = False
batch_size = 4
redundancy_removal = False
balanced_sampler = True

# Experiment name
exp_name="RNA_ligand_"+split+"_"+str(nb_layers)+"layers_lr"+str(learning_rate)+"_"+str(epochs)+"epochs"
if rna_fm:
        exp_name += "_rna_fm"
if redundancy_removal:
        exp_name += "_redundancy_removal"
else:
        exp_name += "_no_redundancy_removal"
if balanced_sampler:
        exp_name += "_balanced_sampler"

# Setup task
class CustomLigandIdentification(LigandIdentification):
        def __init__(self, redundancy_removal, **kwargs):
                self.redundancy_removal = redundancy_removal
                super().__init__(**kwargs)
        def post_process(self):
                """The task-specific post processing steps to remove redundancy and compute distances which will be used by the splitters.
                """
                cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
                if self.redundancy_removal:
                        prepare_dataset = PrepareDataset(distance_name="cd_hit", threshold=0.9)
                us_align_computer = StructureDistanceComputer(name="USalign")
                self.dataset = cd_hit_computer(self.dataset)
                if self.redundancy_removal:
                        self.dataset = prepare_dataset(self.dataset)
                self.dataset = us_align_computer(self.dataset)        

ta_ligand = CustomLigandIdentification(use_balanced_sampler=balanced_sampler,root="roots/RNA_Ligand", redundancy_removal=redundancy_removal)

if split=="struc":
        distance = "USalign"
else:
        distance = "cd_hit"

if distance not in ta_ligand.dataset.distances:
        if split == 'struc':
                ta_ligand.dataset = StructureDistanceComputer()(ta_ligand.dataset)
        if split == 'seq':
                ta_ligand.dataset = CDHitComputer()(ta_ligand.dataset)

if split == 'rand':
        ta_ligand.splitter = RandomSplitter()
else:
        ta_ligand.splitter = ClusterSplitter(distance_name=distance)

if rna_fm:
        rnafm = RNAFMTransform()
        [rnafm(rna) for rna in ta_ligand.dataset]
        ta_ligand.dataset.features_computer.add_feature(feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)})
            
rep = GraphRepresentation(framework="pyg")
ta_ligand.dataset.add_representation(rep)
ta_ligand.get_split_loaders(batch_size=batch_size, recompute=True)

model_ligand_args = {
        "num_node_features": ta_ligand.metadata["num_node_features"],
        "num_classes": ta_ligand.metadata["num_classes"],
        "graph_level": True,
        "num_layers": nb_layers,
}

model_ligand = PygModel(**model_ligand_args)
trainer_ligand = RNATrainer(ta_ligand, model_ligand, rep, exp_name=exp_name, learning_rate=learning_rate, epochs=epochs)
trainer_ligand.train()

if __name__ == "__main__":
        for seed in [0,1,2]:
                model_ligand = PygModel(**model_ligand_args)
                trainer_ligand = RNATrainer(ta_ligand, model_ligand, rep, exp_name=exp_name, learning_rate=learning_rate, epochs=epochs, seed=seed)
                trainer_ligand.train()