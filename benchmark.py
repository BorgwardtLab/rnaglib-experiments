import os
import sys
import copy
from joblib import Parallel, delayed
import time
from rnaglib.learning import PygModel
from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms import RNAFMTransform
from rnaglib.encoders import ListEncoder


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from base.RNA_CM_exp import ta_CM_struc, trainer_CM, model_CM
from base.RNA_GO_exp import ta_GO_struc, trainer_GO, model_GO
from base.RNA_Ligand_exp import ta_ligand_struc, trainer_ligand, model_ligand
from base.RNA_PROT_exp import ta_RBP_struc, trainer_RBP, model_RBP
from base.RNA_SITE_exp import ta_SITE_struc, trainer_SITE, model_SITE
from base.RNA_IF_exp import ta_IF_struc, trainer_IF, model_IF


def do_one(model, task, trainer, seed):
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"Training time: {end - start} seconds")
    print("Trained")


def benchmark():
    TASKLIST = [
        #(ta_CM_struc, ta_CM_seq, model_CM),
        # (ta_GO_struc, ta_GO_seq, model_GO),
        # (ta_ligand_struc, ta_ligand_seq, model_ligand),
        #(ta_RBP_struc, ta_RBP_seq, model_RBP),
        (ta_SITE_struc, ta_SITE_seq, model_SITE),
        #(ta_IF_struc, ta_IF_seq, model_IF),
    ]
    for task_struc, task_seq, model in TASKLIST:
        todo = []
        for task in (task_struc, task_seq):
            print(task.name)
            for use_rnafm in [True, False]:
                if use_rnafm and not task.name == "rna_if":
                    rnafm = RNAFMTransform()
                    [rnafm(rna) for rna in task.dataset]
                    task.dataset.features_computer.add_feature(
                        feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)}
                    )
                else:
                    task.dataset.features_computer.remove_feature(feature_name="rnafm", input_feature=True)
                task.set_loaders(recompute=False)
                for num_layers, model in enumerate(model):
                    for seed in [0, 1, 2]:
                        task_ = copy.deepcopy(task)
                        todo.append((PygModel(**model), num_layers, task_, use_rnafm, seed, task_.root.split("_")[-1]))

            _ = Parallel(n_jobs=-1)(delayed(do_one)(*run_args) for run_args in todo)

def simple_benchmark():
    TASKLIST = [
        #(ta_GO_struc, trainer_GO, model_GO),
        (ta_CM_struc, trainer_CM, model_CM),
        #(ta_ligand_struc, trainer_ligand, model_ligand),
        #(ta_RBP_struc, trainer_RBP, model_RBP),
        #(ta_SITE_struc, trainer_SITE, model_SITE),
        #(ta_IF_struc, trainer_IF, model_IF),
    ]
    for task, trainer, model in TASKLIST:
        todo = []
        print(task.name)
        task.set_loaders(recompute=False)
        for seed in [0, 1, 2]:
            task_ = copy.deepcopy(task)
            todo.append((PygModel(**model), task_, trainer, seed))

        _ = Parallel(n_jobs=-1)(delayed(do_one)(*run_args) for run_args in todo)


if __name__ == "__main__":
    simple_benchmark()
