import json
import os
import random
import time

import numpy as np
import torch
from torch import optim

from rnaglib.encoders import ListEncoder
from rnaglib.tasks.RNA_VS.model import Decoder, LigandGraphEncoder, RNAEncoder, VSModel
from rnaglib.tasks.RNA_VS.task import VirtualScreening
from rnaglib.transforms import FeaturesComputer, GraphRepresentation, RNAFMTransform


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(seed, use_rnafm=False):
    """Training loop with specified random seed and RNAFM option"""
    set_seed(seed)
    print(f"\nStarting training with seed {seed} {'with' if use_rnafm else 'without'} RNAFM")

    # Create a task
    ef_task = VirtualScreening("RNA_VS")

    # Build corresponding datasets and dataloader
    features_computer = FeaturesComputer(nt_features=["nt_code"])
    representations = [GraphRepresentation(framework="dgl")]
    rna_dataset_args = {"representations": representations, "features_computer": features_computer}
    rna_loader_args = {"batch_size": 16, "shuffle": True, "num_workers": 0}

    train_dataset, val_dataset, test_dataset = ef_task.get_split_datasets(dataset_kwargs=rna_dataset_args)

    if use_rnafm:
        print("Adding RNAFM features...")
        rnafm = RNAFMTransform()
        [rnafm(rna) for rna in train_dataset.rna_dataset]
        [rnafm(rna) for rna in val_dataset.rna_dataset]
        [rnafm(rna) for rna in test_dataset.rna_dataset]

        train_dataset.rna_dataset.features_computer.add_feature(
            feature_names=["rnafm"],
            custom_encoders={"rnafm": ListEncoder(640)},
        )
        val_dataset.rna_dataset.features_computer.add_feature(
            feature_names=["rnafm"],
            custom_encoders={"rnafm": ListEncoder(640)},
        )
        test_dataset.rna_dataset.features_computer.add_feature(
            feature_names=["rnafm"],
            custom_encoders={"rnafm": ListEncoder(640)},
        )

    train_dataloader, val_dataloader, test_dataloader = ef_task.get_split_loaders(
        dataset_kwargs=rna_dataset_args,
        dataloader_kwargs=rna_loader_args,
    )

    # Create model
    input_dim = 644 if use_rnafm else 4  # Adjust input dimension based on whether RNAFM is used
    model = VSModel(
        encoder=RNAEncoder(in_dim=input_dim),
        lig_encoder=LigandGraphEncoder(),
        decoder=Decoder(),
    )

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    epochs = 10

    # Training loop
    t0 = time.time()
    for k in range(epochs):
        model.train()
        epoch_losses = []

        for i, batch in enumerate(train_dataloader):
            pockets = batch["pocket"]
            ligands = batch["ligand"]
            actives = torch.tensor(batch["active"], dtype=torch.float32)

            optimizer.zero_grad()
            out = model(pockets, ligands)
            loss = criterion(input=torch.flatten(out), target=actives)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if not i % 5:
                print(
                    f"Epoch {k}, batch {i}/{len(train_dataloader)}, "
                    f"loss: {loss.item():.4f}, time: {time.time() - t0:.1f}s",
                )

        print(f"Epoch {k} completed. Average loss: {np.mean(epoch_losses):.4f}")

    # Final evaluation
    model = model.eval()
    efs = ef_task.evaluate(model)
    return np.mean(efs)


def main():
    seeds = [42, 123, 456]
    rnafm_options = [False, True]
    results = {"with_rnafm": [], "without_rnafm": []}

    for use_rnafm in rnafm_options:
        print(f"\n{'=' * 50}")
        print(f"Running with RNAFM: {use_rnafm}")
        print(f"{'=' * 50}")

        for seed in seeds:
            mean_ef = train_model(seed, use_rnafm)
            key = "with_rnafm" if use_rnafm else "without_rnafm"
            results[key].append(mean_ef)
            print(f"\nResults for seed {seed} ({'with' if use_rnafm else 'without'} RNAFM):")
            print(f"Mean EF: {mean_ef:.4f}")

    # Prepare results dictionary
    results_dict = {
        "per_seed_results": {
            "with_rnafm": {},
            "without_rnafm": {},
        },
        "overall_results": {
            "with_rnafm": {},
            "without_rnafm": {},
        },
    }

    # Store per-seed results
    for i, seed in enumerate(seeds):
        results_dict["per_seed_results"]["with_rnafm"][str(seed)] = {
            "mean_ef": float(results["with_rnafm"][i]),
        }
        results_dict["per_seed_results"]["without_rnafm"][str(seed)] = {
            "mean_ef": float(results["without_rnafm"][i]),
        }

    # Calculate overall statistics for both versions
    for version in ["with_rnafm", "without_rnafm"]:
        mean_ef = float(np.mean(results[version]))
        std_ef = float(np.std(results[version]))
        results_dict["overall_results"][version] = {
            "mean_ef": mean_ef,
            "std_ef": std_ef,
            "seeds": seeds,
            "n_seeds": len(seeds),
        }
        print(f"\nOverall performance {version}:")
        print(f"Mean EF: {mean_ef:.4f} ± {std_ef:.4f}")

    # Create results directory and save
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(results_dir, f"vs_results_{timestamp}.json")

    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()