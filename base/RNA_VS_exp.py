import json
import os
import time
from datetime import datetime

import torch
import wandb
from torch import optim

from rnaglib.tasks.RNA_VS.model import Decoder, LigandGraphEncoder, RNAEncoder, VSModel
from rnaglib.tasks.RNA_VS.task import VirtualScreening
from rnaglib.transforms import FeaturesComputer, GraphRepresentation

# Initialize wandb
exp_name = f"rna_vs_{datetime.now().strftime('%Y%m%d_%H%M')}"
wandb.init(
    entity="mlsb",
    project="rna_virtual_screening",
    name=exp_name,
)

# Create a task
ef_task = VirtualScreening("RNA_VS")

# Build corresponding datasets and dataloader
features_computer = FeaturesComputer(nt_features=["nt_code"])
representations = [GraphRepresentation(framework="dgl")]
rna_dataset_args = {"representations": representations, "features_computer": features_computer}
rna_loader_args = {"batch_size": 16, "shuffle": True, "num_workers": 0}

train_dataloader, val_dataloader, test_dataloader = ef_task.get_split_loaders(
    dataset_kwargs=rna_dataset_args,
    dataloader_kwargs=rna_loader_args,
)

# Create an encoding model
model = VSModel(encoder=RNAEncoder(), lig_encoder=LigandGraphEncoder(), decoder=Decoder())
assert hasattr(model, "predict_ligands") and callable(model.predict_ligands)

# Training setup
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()
epochs = 10

# Training loop with logging
training_log = []
t0 = time.time()

for k in range(epochs):
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
                f"Epoch {k}, batch {i}/{len(train_dataloader)}, loss: {loss.item():.4f}, time: {time.time() - t0:.1f}s"
            )

            # Log batch metrics
            wandb.log(
                {
                    "epoch": k,
                    "batch": i,
                    "batch_loss": loss.item(),
                    "time": time.time() - t0,
                }
            )

    # Calculate and log epoch metrics
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    metrics = {
        "epoch": k,
        "train_loss": epoch_loss,
    }
    wandb.log(metrics)
    training_log.append(metrics)

# Final evaluation
model = model.eval()
final_vs = ef_task.evaluate(model)

# Log final results
results = {
    "test_metrics": final_vs,
    "training_history": training_log,
    "hyperparameters": {
        "learning_rate": learning_rate,
        "batch_size": rna_loader_args["batch_size"],
        "epochs": epochs,
        "nt_features": features_computer.nt_features,
    },
}

# Save results
os.makedirs("results", exist_ok=True)
with open(f"results/{exp_name}_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print final metrics
print("\nFinal Virtual Screening Results:")
for k, v in final_vs.items():
    print(f"{k}: {v:.4f}")
    wandb.run.summary[f"final_{k}"] = v

wandb.finish()
