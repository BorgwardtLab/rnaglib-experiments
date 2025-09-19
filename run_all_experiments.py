import os
import sys
import copy
from joblib import Parallel, delayed

import time

from rnaglib.learning import PygModel, GVPModel
from rnaglib.transforms import GraphRepresentation, SequenceRepresentation, RNAFMTransform, GVPGraphRepresentation
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer, RandomSplitter
from rnaglib.encoders import ListEncoder
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.tasks import get_task

from utils import set_seed
from exp import RNATrainer
from model_seq import SequenceModel
from utils import CustomLigandIdentification
from constants import BEST_HPARAMS, TASKLIST, REPRESENTATIONS, SPLITS, SEEDS


def run_experiment(task, split, seeds = SEEDS, shuffle = False, hparams_dict = None, rna_fm = False, representation = "2.5D", output = "wandb", project_name = "final_benchmark", debug = False):
    start = time.time()
    if hparams_dict is None:
        hparams_dict = BEST_HPARAMS[task][representation][split]

    if hparams_dict == BEST_HPARAMS[task][representation][split]:
        exp_name = f"""{task}_{split}_{representation}{"_rna_fm" if rna_fm else ""}_best_params"""
    else:
        exp_name = f"""{task}_{hparams_dict['num_layers']}layers_lr{hparams_dict['learning_rate']}_{hparams_dict['epochs']}epochs_hiddendim
        {hparams_dict['hidden_channels']}_{representation}_layer_type_rgcn{"_rna_fm" if rna_fm else ""}_{split}"""

    # Instantiate representation
    if representation in ["2D", "2D_GCN"]:
        edge_map = GRAPH_KEYS["2D_edge_map"][TOOL]
    elif representation == "simplified_2.5D":
        edge_map = GRAPH_KEYS["simplified_edge_map"][TOOL]
    elif representation == "2.5D":
        edge_map = GRAPH_KEYS["edge_map"][TOOL]
    
    if representation in ["2D", "2D_GCN", "simplified_2.5D", "2.5D"]:
        graph_representation_args = {
            "framework": "pyg",
            "edge_map": edge_map,
        }
        rep = GraphRepresentation(**graph_representation_args)
    elif representation in ["GVP", "GVP_2.5D"]:
        if representation == "GVP":
            graph_construction = "knn"
            edge_scalar_features = ["RBF"]
        elif representation == "GVP_2.5D":
            graph_construction = "base_pair"
            edge_scalar_features = ["RBF", "LW"]
        gvp_representation_args = {
            "graph_construction": graph_construction,
            "node_scalar_features": ["nt_code"],
            "edge_scalar_features": edge_scalar_features,
        }
        if rna_fm:
            gvp_representation_args["node_scalar_features"].append("rnafm")
        rep = GVPGraphRepresentation(**gvp_representation_args)
    elif representation == "seq":
        rep = SequenceRepresentation(framework="pyg")
    else:
        raise ValueError(f"Representation {representation} not supported")

    # Instantiate task
    remove_redundancy = not task.endswith("redundant")

    task_args = {
        "root": f"roots/{task}",
        "redundancy_removal": remove_redundancy,
        "precomputed": remove_redundancy,
    }
    if task != "rna_ligand":
        ta = get_task(task_id=task.split("_redundant")[0], **task_args)
    else:
        ta = CustomLigandIdentification(**task_args)

    # Set the specified task splitters
    distance = "USalign" if split == "struc" else "cd_hit"
    if distance not in ta.dataset.distances:
        if split == 'struc':
            ta.dataset = StructureDistanceComputer()(ta.dataset)
        if split == 'seq':
            ta.dataset = CDHitComputer()(ta.dataset)
    if split == 'rand':
        ta.splitter = RandomSplitter()
    elif split == 'struc' or split == 'seq':
        if task == "rna_go":
            ta.splitter = ClusterSplitter(similarity_threshold=0.6, distance_name = distance)
        else:
            ta.splitter = ClusterSplitter(distance_name = distance)

    # If the sequence information is needed, add it to the task dataset
    if rna_fm or representation == "seq":
        rnafm = RNAFMTransform()
        [rnafm(rna) for rna in ta.dataset]
        ta.dataset.features_computer.add_feature(feature_names = ["rnafm"], custom_encoders = {"rnafm": ListEncoder(640)})

    # Add the right representation to the task dataset
    ta.dataset.add_representation(rep)

    # Prepare model arguments
    if representation in ["2D", "2D_GCN", "simplified_2.5D", "2.5D"]:
        layer_type = "gcn" if representation == "2D_GCN" else "rgcn"
        model_args = {
            "num_layers": hparams_dict['num_layers'],
            "hidden_channels": hparams_dict['hidden_channels'],
            "dropout_rate": hparams_dict["dropout_rate"],
            "layer_type": layer_type
        }
        if rna_fm:
            model_args["num_node_features"] = 644
    elif representation in ["GVP", "GVP_2.5D"]:
        # Get the dimensions of the (scalar and vector, node and edge) features
        node_s, node_v = ta.dataset[0]['gvp_graph'].h_V
        edge_s, edge_v = ta.dataset[0]['gvp_graph'].h_E
        model_args = {
            "num_classes": ta.metadata["num_classes"],
            "graph_level": ta.metadata["graph_level"],
            "multi_label": ta.metadata["multi_label"],
            "num_layers": hparams_dict['num_layers'],
            "node_in_dim": (node_s.shape[1],node_v.shape[1]),
            "node_h_dim": hparams_dict['h_node_dim'],
            "edge_in_dim": (edge_s.shape[1],edge_v.shape[1]),
            "edge_h_dim": hparams_dict['h_edge_dim'],
            "dropout_rate": hparams_dict["dropout_rate"],
        }
    
    for seed in seeds:

        set_seed(seed)

        # Recompute loaders to change them for each seed (if shuffle = True)
        ta.get_split_loaders(batch_size = hparams_dict["batch_size"], recompute = True, shuffle = shuffle)

        # Re-initialize model dor each seed
        
        if representation in ["2D", "2D_GCN", "simplified_2.5D", "2.5D"]:
            model = PygModel.from_task(ta, **model_args)
        elif representation == "seq":
            model = SequenceModel.from_task(ta, **model_args, num_node_features = 644)
        elif representation in ["GVP", "GVP_2.5D"]:
            model = GVPModel(**model_args)
        else:
            raise ValueError(f"Representation {representation} not supported")
        
        if debug:
            epochs = 1
        else:
            epochs = hparams_dict["epochs"]
        trainer = RNATrainer(
            ta, 
            model, 
            rep, 
            wandb_project = project_name,
            exp_name = f"{exp_name}_seed{seed}", 
            learning_rate = hparams_dict["learning_rate"], 
            epochs = epochs,
            seed = seed, 
            batch_size = hparams_dict["batch_size"], 
            output = output,
            loss_weights=hparams_dict["loss_weights"],
        )
        trainer.train()
        end = time.time()
        print(f"Training time: {end - start} seconds")
        print("Trained")

if __name__ == "__main__":
    hparams_dict = BEST_HPARAMS
    rna_fm = False
    output = "wandb"
    
    for task in TASKLIST:
        task_params = []
        for representation in hparams_dict[task]:
            if representation in REPRESENTATIONS:
                for split in hparams_dict[task][representation]:
                    if split in SPLITS:
                        params = (task, split, SEEDS, True, hparams_dict[task][representation][split], rna_fm, representation, output, "final_benchmark", False)
                        task_params.append(params)
        _ = Parallel(n_jobs=-1)(delayed(run_experiment)(*params) for params in task_params)