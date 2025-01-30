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
from base.RNA_CM_exp import ta_CM_struc, ta_CM_seq, models_CM
from base.RNA_GO_exp import ta_GO_struc, ta_GO_seq, models_GO
from base.RNA_Ligand_exp import ta_ligand_struc, ta_ligand_seq, models_ligand
from base.RNA_PROT_exp import ta_RBP_struc, ta_RBP_seq, models_RBP
from base.RNA_SITE_exp import ta_SITE_struc, ta_SITE_seq, models_SITE
from base.RNA_IF_exp import ta_IF_struc, ta_IF_seq, models_IF


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
        (ta_CM_struc, ta_CM_seq, models_CM),
        (ta_GO_struc, ta_GO_seq, models_GO),
        (ta_ligand_struc, ta_ligand_seq, models_ligand),
        (ta_RBP_struc, ta_RBP_seq, models_RBP),
        (ta_SITE_struc, ta_SITE_seq, models_SITE),
        (ta_IF_struc, ta_IF_seq, models_IF),
    ]
    for (task_struc, task_seq), models in TASKLIST:
        todo = []
        for task in (task_struc, task_seq):
            print(task.name)
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
                        todo.append((PygModel(**model), num_layers, task_, use_rnafm, seed, task_.root))

            _ = Parallel(n_jobs=-1)(delayed(do_one)(*run_args) for run_args in todo)


if __name__ == "__main__":
    benchmark()
