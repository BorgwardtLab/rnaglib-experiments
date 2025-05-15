import os
import sys
import json
from datetime import datetime
import wandb

import torch
import numpy as np

class RNATrainer:
    def __init__(self, task, model, rep="pyg", wandb_project="", exp_name="default",
                 learning_rate=0.001, epochs=100, seed=0, batch_size=8):
        self.task = task
        self.representation = rep
        self.model = model
        self.wandb_project = wandb_project
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.exp_name = exp_name
        self.training_log = []
        self.seed = seed
        self.batch_size = batch_size

    def setup(self):
        """Initialize wandb and model training"""
        wandb.init(
            entity="mlsb",  # Replace with your team name
            project=self.wandb_project,
            name=self.exp_name,
        )
    
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)  # CPU random number generator
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)  # GPU random number generator for all GPUs

        # For additional control, especially with PyTorch's cudnn backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model.configure_training(learning_rate=self.learning_rate)

        self.task.add_representation(self.representation)

        self.task.get_split_loaders(recompute=False,
                                    batch_size=self.batch_size)

    def train(self):
        """Run training loop with logging"""
        self.setup()

        if self.model.num_classes == 2:
            neg_count = float(self.task.metadata["class_distribution"]["0"])
            pos_count = float(self.task.metadata["class_distribution"]["1"])
            pos_weight = torch.tensor(np.sqrt(neg_count /
                                              pos_count)).to(self.model.device,
                                                             dtype=torch.float32)
            self.model.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            for batch in self.task.train_dataloader:
                graph = batch[self.representation.name].to(self.model.device)
                self.model.optimizer.zero_grad()
                out = self.model(graph)
                loss = self.model.compute_loss(out, graph.y)
                loss.backward()
                self.model.optimizer.step()

            # Evaluation phase
            train_metrics = self.model.evaluate(self.task, split="train")
            val_metrics = self.model.evaluate(self.task, split="val")

            # Log to wandb
            metrics = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            try:
                metrics["train_auc"] = train_metrics['auc']
                metrics["val_auc"] = val_metrics['auc']
            except:
                pass
            if self.task.metadata['multi_label']:
                metrics["train_jaccard"] = train_metrics["jaccard"]
                metrics["val_jaccard"] = val_metrics["jaccard"]
            else:
                try:
                    metrics["train_balanced_accuracy"] = train_metrics["balanced_accuracy"]
                    metrics["val_balanced_accuracy"] = val_metrics["balanced_accuracy"]
                except:
                    pass
                try:
                    metrics["train_mcc"] = train_metrics["mcc"]
                    metrics["val_mcc"] = val_metrics["mcc"]
                except:
                    pass

            wandb.log(metrics)
            self.training_log.append(metrics)

            # Print progress
            if not epoch % 20:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}",
                )

        self.save_results()
        wandb.finish()

    def save_results(self):
        """Save final results and metrics"""
        # Final evaluation
        test_metrics = self.model.evaluate(self.task)

        # Get detailed predictions
        _, all_preds, all_probs, all_labels = self.model.inference(self.task.test_dataloader)

        # Prepare results
        results = {
            "test_metrics": test_metrics,
            "training_history": self.training_log,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "num_node_features": self.model.num_node_features,
                "num_classes": self.model.num_classes,
                "graph_level": self.model.graph_level,
                "num_layers": self.model.num_layers,
                "hidden_channels": self.model.hidden_channels,
                "dropout_rate": self.model.dropout_rate,
                "multi_label": self.model.multi_label,
            },
        }

        # Save to file
        os.makedirs("results", exist_ok=True)
        with open(f"results/{self.exp_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print final metrics
        print("\nFinal Test Results:")
        for k, v in test_metrics.items():
            print(k, v)
            # print(f"Test {k}: {v:.4f}")


# Example usage:
if __name__ == "__main__":
    from rnaglib.learning.task_models import PygModel
    from rnaglib.tasks import BindingSite
    from rnaglib.transforms import GraphRepresentation

    # Setup task
    ta = BindingSite(root="RNA_Site", debug=True, recompute=True)
    ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta.get_split_loaders(recompute=True)

    # Create model
    model = PygModel(
        ta.metadata["description"]["num_node_features"],
        ta.metadata["description"]["num_classes"],
        graph_level=False,
    )

    # Create trainer and run
    trainer = RNATrainer(ta, model, wandb_project="rna_binding_site")
    trainer.train()
