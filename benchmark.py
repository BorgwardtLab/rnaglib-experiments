import os
import sys
from joblib import Parallel, delayed

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
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


def do_one(model, num_layers, task, distance, rnafm, seed):
    task.dataset = distance(task.dataset)
    task.splitter = ClusterSplitter(distance_name=distance.name)

    task.dataset.add_representation(GraphRepresentation(framework="pyg"))
    task.get_split_loaders(recompute=True)
    if use_rnafm:
        rnafm = RNAFMTransform()
        [rnafm(rna) for rna in task.dataset]
        task.dataset.features_computer.add_feature(feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)})
        print("Done computing embs")
        trainer = RNATrainer(
            task,
            model,
            exp_name=f"{task.name}_rnafm-{rnafm}_distance-{distance.name}_layers-{num_layers}_seed-{seed}",
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
    TODO = []
    for task, models in TASKLIST:
        for distance in [CDHitComputer(), StructureDistanceComputer()]:
            for rnafm in [True, False]:
                for num_layers, model in enumerate(models):
                    for seed in [0, 1, 2]:
                        TODO.append((model, num_layers, task, distance, rnafm, seed))

    _ = Parallel(n_jobs=-1)(delayed(do_one)(*run_args) for run_args in TODO)


if __name__ == "__main__":
    benchmark()
