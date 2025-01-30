import os
import sys

from joblib import Parallel, delayed

from rnaglib.encoders import ListEncoder
from rnaglib.transforms import RNAFMTransform

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from base.RNA_SITE_bm_exp import models_SITE_bm, ta_SITE_bm
from exp import RNATrainer


def do_one(model, num_layers, task, use_rnafm, seed):
    print("Done computing embs")
    trainer = RNATrainer(
        task,
        model,
        exp_name=f"{task.name}_rnafm-{use_rnafm}_layers-{num_layers}_seed-{seed}",
        seed=seed,
    )
    print("Training")
    trainer.train()
    print("Trained")


def benchmark():
    TASKLIST = [
        (ta_SITE_bm, models_SITE_bm),
    ]
    for task, models in TASKLIST:
        todo = []
        for use_rnafm in [True, False]:
            if use_rnafm:
                rnafm = RNAFMTransform()
                [rnafm(rna) for rna in task.dataset]
                task.dataset.features_computer.add_feature(
                    feature_names=["rnafm"],
                    custom_encoders={"rnafm": ListEncoder(640)},
                )
            for num_layers, model in enumerate(models):
                for seed in [0, 1, 2]:
                    todo.append((model, num_layers, task, use_rnafm, seed))

        _ = Parallel(n_jobs=-1)(delayed(do_one)(*run_args) for run_args in todo)


if __name__ == "__main__":
    benchmark()
