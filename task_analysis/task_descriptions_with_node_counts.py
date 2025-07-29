#!/usr/bin/env python3
"""
Script to load all tasks, count actual nodes in RNA graphs, 
get task descriptions, and write everything to a CSV file.
"""

import csv
import os
import sys
from datetime import datetime
from collections import defaultdict
from rnaglib.tasks import get_task


# List of all available tasks
TASKS = ["rna_cm",
         "rna_go", 
         "rna_if",
         "rna_if_bench",
         "rna_ligand",
         "rna_prot",
         "rna_site",
         "rna_site_bench"]


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary structure.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested structure
        sep: Separator for nested keys
        
    Returns:
        dict: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert lists/tuples to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def count_nodes_in_task(task_name):
    """
    Load a task and count the total number of nodes across all RNA graphs.
    
    Args:
        task_name (str): Name of the task to analyze
        
    Returns:
        tuple: (total_nodes, task_description, dataset_size)
    """
    print(f"\n{'='*50}")
    print(f"Loading and counting nodes for task: {task_name}")
    print(f"{'='*50}")
    
    try:
        # Load the task with specific root directory
        root_dir = f"./analyser_roots/{task_name}"
        print(f"Using root directory: {root_dir}")
        ta = get_task(task_id=task_name, root=root_dir, precomputed=True)
    
        # Get task description
        task_info = ta.describe()
        print(f"Task description keys: {list(task_info.keys())}")
        
        # Flatten the task description for easier CSV handling
        flattened_task_info = flatten_dict(task_info)
        
        # Count nodes in all RNA graphs
        total_nodes = 0
        dataset_size = len(ta.dataset)
        
        print(f"Counting nodes in {dataset_size} RNA graphs...")
        
        for i in range(dataset_size):
            rna_data = ta.dataset[i]
            rna_graph = rna_data['rna']
            
            # Count nodes in this RNA graph (using NetworkX method)
            num_nodes = len(rna_graph.nodes)
            total_nodes += num_nodes
            
            if i < 5:  # Show details for first 5 graphs
                print(f"  Graph {i+1} ({rna_graph.name}): {num_nodes} nodes")
            elif i == 5:
                print("  ...")
        
        print(f"Total nodes across all {dataset_size} graphs: {total_nodes}")
        print(f"Average nodes per graph: {total_nodes / dataset_size:.2f}")
        
        return total_nodes, flattened_task_info, dataset_size
        
    except Exception as e:
        print(f"Error analyzing task {task_name}: {str(e)}")
        return 0, {}, 0


def main():
    """Main function to process all tasks and write results to CSV."""
    
    print("=" * 60)
    print("TASK DESCRIPTIONS WITH ACTUAL NODE COUNTS")
    print("=" * 60)
    print("Loading tasks and counting nodes in RNA graphs...")
    print()
    
    # Dictionary to store results for each task
    all_task_data = {}
    
    # Process each task
    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"PROCESSING TASK: {task_name}")
        print(f"{'='*60}")
        
        total_nodes, task_description, dataset_size = count_nodes_in_task(task_name)
        
        # Store the data
        all_task_data[task_name] = {
            'total_nodes': total_nodes,
            'dataset_size': dataset_size,
            'task_description': task_description
        }
    
    print(f"\n{'='*50}")
    print("WRITING RESULTS TO CSV")
    print(f"{'='*50}")
    
    # Collect all unique description fields across all tasks
    all_desc_fields = set()
    for task_data in all_task_data.values():
        all_desc_fields.update(task_data['task_description'].keys())
    
    # Sort fields alphabetically for consistent column order
    sorted_desc_fields = sorted(all_desc_fields)
    
    # Create CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"task_descriptions_with_node_counts_{timestamp}.csv"
    
    print(f"Writing results to {csv_filename}...")
    
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Task_Name', 'Total_Nodes', 'Dataset_Size', 'Avg_Nodes_Per_Graph'] + sorted_desc_fields
            writer.writerow(header)
            
            # Write data for each task
            for task_name in TASKS:
                task_data = all_task_data[task_name]
                
                # Calculate average nodes per graph
                if task_data['dataset_size'] > 0:
                    avg_nodes = task_data['total_nodes'] / task_data['dataset_size']
                else:
                    avg_nodes = 0
                
                # Start building the row
                row = [
                    task_name,
                    task_data['total_nodes'],
                    task_data['dataset_size'],
                    f"{avg_nodes:.2f}"
                ]
                
                # Add values for each description field
                task_desc = task_data['task_description']
                for field in sorted_desc_fields:
                    value = task_desc.get(field, '')
                    row.append(value)
                
                writer.writerow(row)
        
        print(f"Successfully written to {csv_filename}")
        print(f"CSV structure: {len(TASKS)} rows (tasks) × {len(header)} columns")
        
        # Print summary
        print(f"\nSUMMARY:")
        print("=" * 60)
        print("Task\t\tDataset Size\tTotal Nodes\tAvg Nodes/Graph")
        print("-" * 60)
        
        total_graphs = 0
        total_all_nodes = 0
        
        for task_name in TASKS:
            task_data = all_task_data[task_name]
            dataset_size = task_data['dataset_size']
            total_nodes = task_data['total_nodes']
            avg_nodes = total_nodes / dataset_size if dataset_size > 0 else 0
            
            total_graphs += dataset_size
            total_all_nodes += total_nodes
            
            print(f"{task_name:<15}\t{dataset_size:<10}\t{total_nodes:<12}\t{avg_nodes:.2f}")
        
        overall_avg = total_all_nodes / total_graphs if total_graphs > 0 else 0
        print("-" * 60)
        print(f"{'TOTAL':<15}\t{total_graphs:<10}\t{total_all_nodes:<12}\t{overall_avg:.2f}")
        
        print(f"\n✅ Results saved to: {csv_filename}")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 