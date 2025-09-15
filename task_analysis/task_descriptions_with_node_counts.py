#!/usr/bin/env python3
"""
Script to load all tasks, analyze RNA graph node statistics (including min, max, 
median, mean sizes and counts/percentages of large RNAs), get task descriptions, 
and write everything to a CSV file.
"""

import csv
import os
import sys
import statistics
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
    Load a task and analyze node counts across all RNA graphs.
    
    Args:
        task_name (str): Name of the task to analyze
        
    Returns:
        tuple: (total_nodes, task_description, dataset_size, node_stats)
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
        
        # Count nodes in all RNA graphs and collect individual sizes
        total_nodes = 0
        node_counts = []  # Store individual node counts for statistics
        dataset_size = len(ta.dataset)
        
        print(f"Counting nodes in {dataset_size} RNA graphs...")
        
        for i in range(dataset_size):
            rna_data = ta.dataset[i]
            rna_graph = rna_data['rna']
            
            # Count nodes in this RNA graph (using NetworkX method)
            num_nodes = len(rna_graph.nodes)
            total_nodes += num_nodes
            node_counts.append(num_nodes)
            
            if i < 5:  # Show details for first 5 graphs
                print(f"  Graph {i+1} ({rna_graph.name}): {num_nodes} nodes")
            elif i == 5:
                print("  ...")
        
        # Calculate additional statistics
        if node_counts:
            min_nodes = min(node_counts)
            max_nodes = max(node_counts)
            mean_nodes = total_nodes / dataset_size
            median_nodes = statistics.median(node_counts)
            
            # Count RNAs with 200+ nodes
            large_rnas_count = sum(1 for count in node_counts if count >= 200)
            large_rnas_percentage = (large_rnas_count / dataset_size) * 100 if dataset_size > 0 else 0
        else:
            min_nodes = max_nodes = mean_nodes = median_nodes = 0
            large_rnas_count = 0
            large_rnas_percentage = 0
        
        node_stats = {
            'min_nodes': min_nodes,
            'max_nodes': max_nodes,
            'mean_nodes': mean_nodes,
            'median_nodes': median_nodes,
            'large_rnas_count': large_rnas_count,
            'large_rnas_percentage': large_rnas_percentage
        }
        
        print(f"Total nodes across all {dataset_size} graphs: {total_nodes}")
        print(f"Average nodes per graph: {mean_nodes:.2f}")
        print(f"Min nodes: {min_nodes}, Max nodes: {max_nodes}, Median nodes: {median_nodes:.2f}")
        print(f"RNAs with ≥200 nodes: {large_rnas_count} ({large_rnas_percentage:.1f}%)")
        
        return total_nodes, flattened_task_info, dataset_size, node_stats
        
    except Exception as e:
        print(f"Error analyzing task {task_name}: {str(e)}")
        # Return empty node_stats dict for consistency
        empty_node_stats = {
            'min_nodes': 0,
            'max_nodes': 0,
            'mean_nodes': 0,
            'median_nodes': 0,
            'large_rnas_count': 0,
            'large_rnas_percentage': 0
        }
        return 0, {}, 0, empty_node_stats


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
        
        total_nodes, task_description, dataset_size, node_stats = count_nodes_in_task(task_name)
        
        # Store the data
        all_task_data[task_name] = {
            'total_nodes': total_nodes,
            'dataset_size': dataset_size,
            'task_description': task_description,
            'node_stats': node_stats
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
            
            # Write header including new statistics columns
            stats_columns = ['Min_Nodes', 'Max_Nodes', 'Mean_Nodes', 'Median_Nodes', 
                           'Large_RNAs_Count_200plus', 'Large_RNAs_Percentage_200plus']
            header = ['Task_Name', 'Total_Nodes', 'Dataset_Size'] + stats_columns + sorted_desc_fields
            writer.writerow(header)
            
            # Write data for each task
            for task_name in TASKS:
                task_data = all_task_data[task_name]
                node_stats = task_data['node_stats']
                
                # Start building the row
                row = [
                    task_name,
                    task_data['total_nodes'],
                    task_data['dataset_size'],
                    f"{node_stats['min_nodes']:.0f}",
                    f"{node_stats['max_nodes']:.0f}",
                    f"{node_stats['mean_nodes']:.2f}",
                    f"{node_stats['median_nodes']:.2f}",
                    f"{node_stats['large_rnas_count']:.0f}",
                    f"{node_stats['large_rnas_percentage']:.1f}"
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
        print("=" * 120)
        print("Task\t\tDataset\tTotal\tMin\tMax\tMean\tMedian\t≥200\t≥200%")
        print("\t\tSize\tNodes\tNodes\tNodes\tNodes\tNodes\tCount\t")
        print("-" * 120)
        
        total_graphs = 0
        total_all_nodes = 0
        total_large_rnas = 0
        
        for task_name in TASKS:
            task_data = all_task_data[task_name]
            dataset_size = task_data['dataset_size']
            total_nodes = task_data['total_nodes']
            node_stats = task_data['node_stats']
            
            total_graphs += dataset_size
            total_all_nodes += total_nodes
            total_large_rnas += node_stats['large_rnas_count']
            
            print(f"{task_name:<15}\t{dataset_size:<7}\t{total_nodes:<7}\t{node_stats['min_nodes']:<4.0f}\t{node_stats['max_nodes']:<5.0f}\t{node_stats['mean_nodes']:<5.1f}\t{node_stats['median_nodes']:<6.1f}\t{node_stats['large_rnas_count']:<5.0f}\t{node_stats['large_rnas_percentage']:<5.1f}%")
        
        overall_avg = total_all_nodes / total_graphs if total_graphs > 0 else 0
        overall_large_pct = (total_large_rnas / total_graphs) * 100 if total_graphs > 0 else 0
        print("-" * 120)
        print(f"{'TOTAL':<15}\t{total_graphs:<7}\t{total_all_nodes:<7}\t{'-':<4}\t{'-':<5}\t{overall_avg:<5.1f}\t{'-':<6}\t{total_large_rnas:<5}\t{overall_large_pct:<5.1f}%")
        
        print(f"\n✅ Results saved to: {csv_filename}")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 