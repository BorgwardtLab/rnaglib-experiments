import os
import sys

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import BindingSite
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from exp import RNATrainer
from base.RNA_CM_exp import ta_CM, model_CM
from base.RNA_GO_exp import ta_CM, model_CM


def benchmark():
    TASKLIST = [(ta_CM, model_CM), (ta_GO, model_GO)]

    for task, model in TASKLIST:
        for use_rnafm in [True, False]:
            for distance in [CDHitComputer(), StructureDistanceComputer()]:
                print("Splitting")
                task.dataset = distance(task.dataset)
                task.splitter = ClusterSplitter(distance_name=distance.name)

                task.dataset.add_representation(GraphRepresentation(framework="pyg"))
                task.get_split_loaders(recompute=False)
                print("Got splits")

                if use_rnafm:
                    rnafm = RNAFMTransform()
                    [rnafm(rna) for rna in task.dataset]
                    task.dataset.features_computer.add_feature(
                        feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)}
                    )
                    print("Done computing embs")
                # Create trainer and run
                trainer = RNATrainer(task, model, wandb_project="rna_binding_site")
                print("Training")
                trainer.train()
                print("Trained")
