"""experiment setup."""

import os
import sys
import shutil
import json
from itertools import product

# Add parent directory to Python path to find the exp module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task 
from rnaglib.transforms import GraphRepresentation

from exp import RNATrainer

ta = get_task(root="roots/RNA_IF", task_id="rna_if")
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.get_split_loaders()

model_args  =   {
        "num_node_features": ta.metadata["num_node_features"],
        "num_classes": ta.metadata["num_classes"] + 1,
        "graph_level": False,
        "num_layers": 3,
    }
model = PygModel(**model_args)
trainer = RNATrainer(ta, model)

trainer.train()


if __name__ == "__main__":
    # Hyperparameters for grid search
    learning_rates = [0.0001, 0.001, 0.01]
    num_layers_options = [2, 3, 4]
    hidden_dims = [32, 64, 128]
    
    # Store results for each combination
    grid_search_results = []
    
    # Setup experiment directory for results
    os.makedirs("results/grid_search_RNA_IF", exist_ok=True)
    
    # Perform grid search
    for lr, n_layers, hidden_dim in product(learning_rates, num_layers_options, hidden_dims):
        print(f"\n--- Training with lr={lr}, num_layers={n_layers}, hidden_dim={hidden_dim} ---\n")
        
        # Create model with current hyperparameters
        current_model_args = {
            "num_node_features": ta.metadata["num_node_features"],
            "num_classes": ta.metadata["num_classes"] + 1,
            "graph_level": False,
            "num_layers": n_layers,
            "hidden_channels": hidden_dim
        }
        
        model = PygModel(**current_model_args)
        
        # Create experiment name based on hyperparameters
        exp_name = f"rna_if_lr{lr}_layers{n_layers}_hidden{hidden_dim}"
        
        # Initialize trainer with current hyperparameters
        trainer = RNATrainer(
            ta, 
            model, 
            exp_name=exp_name,
            learning_rate=lr
        )
        
        # Train the model
        trainer.train()
        
        # Get test metrics for this configuration
        test_metrics = model.evaluate(ta)
        
        # Store results
        result = {
            "hyperparameters": {
                "learning_rate": lr,
                "num_layers": n_layers,
                "hidden_dim": hidden_dim
            },
            "test_metrics": test_metrics
        }
        
        grid_search_results.append(result)
        
        # Save individual result
        with open(f"results/grid_search_RNA_IF/{exp_name}_results.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Find best configuration based on test accuracy
    best_result = max(grid_search_results, key=lambda x: x["test_metrics"]["accuracy"])
    
    # Save all results and best configuration
    with open("results/grid_search_RNA_IF/all_results.json", "w") as f:
        json.dump({
            "all_results": grid_search_results,
            "best_result": best_result
        }, f, indent=2)
    
    print("\n----- Grid Search Complete -----")
    print("Best Configuration:")
    print(f"Learning Rate: {best_result['hyperparameters']['learning_rate']}")
    print(f"Number of Layers: {best_result['hyperparameters']['num_layers']}")
    print(f"Hidden Dimension: {best_result['hyperparameters']['hidden_dim']}")
    print("Test Metrics:")
    for metric, value in best_result["test_metrics"].items():
        print(f"{metric}: {value}")

