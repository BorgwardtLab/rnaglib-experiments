import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from exp import RNATrainer
from base.RNA_CM_exp import ta_CM, models_CM
from base.RNA_GO_exp import ta_GO, models_GO
from base.RNA_Ligand_exp import ta_ligand, models_ligand
from base.RNA_IF_exp import ta_IF, models_IF
from base.RNA_RBP_exp import ta_RBP, models_RBP
from base.RNA_SITE_exp import ta_SITE, models_SITE


def benchmark():
    TASKLIST = [
        (ta_CM, model_CM),
        (ta_GO, model_GO),
        (ta_IF, model_IF),
        (ta_ligand, model_ligand),
        (ta_RBP, model_RBP),
        (ta_SITE, model_SITE),
    ]

    for task, models in TASKLIST:
        for distance in [CDHitComputer(), StructureDistanceComputer()]:
            print("Splitting")
            task.dataset = distance(task.dataset)
            task.splitter = ClusterSplitter(distance_name=distance.name)

            task.dataset.add_representation(GraphRepresentation(framework="pyg"))
            task.get_split_loaders(recompute=False)
            print("Got splits")
            for use_rnafm in [True, False]:
                if use_rnafm:
                    rnafm = RNAFMTransform()
                    [rnafm(rna) for rna in task.dataset]
                    task.dataset.features_computer.add_feature(
                        feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)}
                    )
                    print("Done computing embs")
                for num_layers, model in models:
                    for seed in [0, 1, 2]:
                        trainer = RNATrainer(
                            task,
                            model,
                            exp_name=f"{task.name}_rnafm-{use_rnafm}_distance-{distance.name}_layers-{num_layers}_seed-{seed}",
                            seed=seed,
                        )
                        print("Training")
                        trainer.train()
                        print("Trained")
