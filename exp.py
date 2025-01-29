import os
import sys
import json
from datetime import datetime
import wandb

from rnaglib.transforms import RNAFMTransform
from rnaglib.dataset_transforms import CDHitComputer, StructureDistanceComputer, ClusterSplitter
from rnaglib.encoders import ListEncoder


class RNATrainer:
    def __init__(self, task, model, wandb_project, exp_name="default", learning_rate=0.001, epochs=2, seed=0):
        self.task = task
        self.model = model
        self.wandb_project = wandb_project
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.exp_name = exp_name
        self.training_log = []
        self.seed = seed

    def setup(self):
        """Initialize wandb and model training"""
        """
        wandb.init(
            entity="mlsb",  # Replace with your team name
            project=self.wandb_project,
            name=self.exp_name,
        )
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)  # CPU random number generator
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)  # GPU random number generator for all GPUs

        # For additional control, especially with PyTorch's cudnn backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model.configure_training(learning_rate=self.learning_rate)

    def train(self):
        """Run training loop with logging"""
        self.setup()

        print("Getting split loaders")
        train_loader, _, _ = self.task.get_split_loaders()
        print("Got split loaders")
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            for batch in train_loader:
                graph = batch["graph"].to(self.model.device)
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
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
            }
            # wandb.log(metrics)
            self.training_log.append(metrics)

            # Print progress
            if not epoch % 20:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}",
                )

        self.save_results()
        # wandb.finish()

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
                "num_node_features": self.model.convs[0].in_channels,
                "num_classes": self.model.num_classes,
                "graph_level": self.model.graph_level,
                "num_layers": self.model.num_layers,
                "hidden_channels": self.model.convs[0].out_channels,
                "dropout_rate": self.model.dropouts[0].p,
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
            print(f"Test {k}: {v:.4f}")


# Example usage:
if __name__ == "__main__":
    benchmark()

    """
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
    """
