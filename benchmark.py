import os
import sys
import copy
from joblib import Parallel, delayed

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms import RNAFMTransform
from rnaglib.encoders import ListEncoder
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from exp import RNATrainer
from base.RNA_CM_exp import ta_CM, models_CM
from base.RNA_GO_exp import ta_GO, models_GO
from base.RNA_Ligand_exp import ta_ligand, models_ligand
from base.RNA_IF_exp import ta_IF, models_IF
from base.RNA_PROT_exp import ta_RBP, models_RBP
from base.RNA_SITE_exp import ta_SITE, models_SITE


def do_one(model, num_layers, task, use_rnafm, seed, distance):
    print("Done computing embs")
    task.dataset.add_representation(GraphRepresentation(framework="pyg"))

    # have to re-init the model with different input dim
    if use_rnafm:
        print(model)
        model = PygModel(
            num_node_features=644,
            num_classes=model.num_classes,
            num_unique_edge_attrs=model.num_unique_edge_attrs,
            graph_level=model.graph_level,
            num_layers=model.num_layers,
            hidden_channels=model.hidden_channels,
            dropout_rate=model.dropout_rate,
            multi_label=model.multi_label,
        )
        print(model)

    trainer = RNATrainer(
        task,
        model,
        exp_name=f"{task.name}_rnafm-{use_rnafm}_distance-{distance}_layers-{num_layers}_seed-{seed}",
        seed=seed,
    )
    print("Training")
    trainer.train()
    print("Trained")


def benchmark():
    TASKLIST = [
        (ta_CM, models_CM),
        (ta_GO, models_GO),
        (ta_IF, models_IF),
        (ta_ligand, models_ligand),
        (ta_RBP, models_RBP),
        (ta_SITE, models_SITE),
    ]
    for task, models in TASKLIST:
        print(task.name)
        for distance in [CDHitComputer(), StructureDistanceComputer()]:
            task.dataset = distance(task.dataset)
            task.splitter = ClusterSplitter(distance_name=distance.name)

            task.set_loaders(recompute=True)
            todo = []
            for use_rnafm in [True, False]:
                if use_rnafm:
                    rnafm = RNAFMTransform()
                    [rnafm(rna) for rna in task.dataset]
                    task.dataset.features_computer.add_feature(
                        feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)}
                    )
                else:
                    task.dataset.features_computer.remove_feature(feature_name="rnafm", input_feature=True)
                task.set_loaders(recompute=False)
                for num_layers, model in enumerate(models):
                    for seed in [0, 1, 2]:
                        task_ = copy.deepcopy(task)
                        todo.append((model, num_layers, task_, use_rnafm, seed, distance.name))

            _ = Parallel(n_jobs=-1)(delayed(do_one)(*run_args) for run_args in todo)


if __name__ == "__main__":
    benchmark()
