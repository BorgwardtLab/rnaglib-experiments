#!/usr/bin/env python3
"""
Task analyser script that extracts RFAM family information from RNA data
for all available tasks by looking up PDB IDs in the FURNA dataset.
"""

import csv
import sys
import os
from datetime import datetime
from collections import Counter, defaultdict
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


class TeeOutput:
    """Class to write output to both file and console"""
    def __init__(self, file_obj, console_obj):
        self.file = file_obj
        self.console = console_obj
    
    def write(self, data):
        self.file.write(data)
        self.console.write(data)
        self.file.flush()
        self.console.flush()
    
    def flush(self):
        self.file.flush()
        self.console.flush()


def load_furna_mapping():
    """
    Load FURNA dataset and create a mapping from PDB_ID to RFAM accession.
    
    Returns:
        dict: Dictionary mapping PDB IDs to RFAM accessions
    """
    print("Loading FURNA dataset mapping...")
    furna_path = "data/FURNA_dataset.txt"
    
    if not os.path.exists(furna_path):
        raise FileNotFoundError(f"FURNA dataset not found at {furna_path}")
    
    pdb_to_rfam = {}
    
    with open(furna_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 5:  # Ensure we have at least 5 columns
                    pdb_id = parts[0].strip()
                    chain_id = parts[1].strip()
                    rfam_acc = parts[4].strip()
                    
                    # Create PDB key (some datasets might include chain, some might not)
                    pdb_key = pdb_id.lower()
                    pdb_chain_key = f"{pdb_id.lower()}.{chain_id}".lower()
                    
                    # Only store if RFAM accession starts with RF
                    if rfam_acc.startswith('RF'):
                        pdb_to_rfam[pdb_key] = rfam_acc
                        pdb_to_rfam[pdb_chain_key] = rfam_acc
                        
            except Exception as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(pdb_to_rfam)} PDB-RFAM mappings from FURNA dataset")
    return pdb_to_rfam


def extract_pdb_id(rna_name):
    """
    Extract PDB ID from RNA name in various formats.
    
    Args:
        rna_name (str): RNA name from the dataset
        
    Returns:
        str: Extracted PDB ID in lowercase
    """
    # Handle various naming conventions
    if '.' in rna_name:
        # Format like "1a1t.A" -> "1a1t"
        return rna_name.split('.')[0].lower()
    elif '_' in rna_name:
        # Format like "1a1t_A" -> "1a1t"
        return rna_name.split('_')[0].lower()
    else:
        # Simple PDB ID
        return rna_name.lower()


def analyze_task(task_name, pdb_to_rfam):
    """
    Analyze a single task and return RFAM family counts by looking up PDB IDs in FURNA dataset.
    
    Args:
        task_name (str): Name of the task to analyze
        pdb_to_rfam (dict): Mapping from PDB IDs to RFAM accessions
        
    Returns:
        dict: Counter object with RFAM family counts
    """
    print(f"\n{'='*50}")
    print(f"Analyzing task: {task_name}")
    print(f"{'='*50}")
    
    try:
        # Load the task with specific root directory
        root_dir = f"./analyser_roots/{task_name}"
        print(f"Using root directory: {root_dir}")
        ta = get_task(task_id=task_name, root=root_dir)
    
        # Display task description
        task_info = ta.describe()
        print("Dataset Description:")
        print(f"num_node_features: {task_info['num_node_features']}")
        print(f"num_classes: {task_info['num_classes']}")
        print(f"dataset_size: {task_info['dataset_size']}")
        print("Class distribution:")
        for class_name, count in task_info['class_distribution'].items():
            print(f"\tClass {class_name}: {count} nodes")
        
        print("\nExtracting PDB IDs and looking up RFAM families...")
        
        # Extract PDB IDs and look up RFAM families
        rfam_list = []
        pdb_found = 0
        pdb_not_found = 0
        
        for i in range(len(ta.dataset)):
            rna_data = ta.dataset[i]
            rna_graph = rna_data['rna']
            rna_name = rna_graph.name
            
            # Extract PDB ID
            pdb_id = extract_pdb_id(rna_name)
            
            # Look up in FURNA dataset
            rfam_acc = None
            
            # Try different key formats
            for key in [pdb_id, rna_name.lower()]:
                if key in pdb_to_rfam:
                    rfam_acc = pdb_to_rfam[key]
                    pdb_found += 1
                    break
            
            if rfam_acc is None:
                pdb_not_found += 1
            
            rfam_list.append(rfam_acc)

        print(f"Number of elements: {len(rfam_list)}")
        print(f"PDB IDs found in FURNA: {pdb_found}")
        print(f"PDB IDs not found in FURNA: {pdb_not_found}")
        print(f"First 10 RFAM values: {rfam_list[:10]}")
        print(f"Unique RFAM families: {set(rfam_list)}")
        print(f"Number of unique families: {len(set(rfam_list))}")
        
        # Count occurrences of each RFAM family type (including nulls)
        rfam_counts = Counter(rfam_list)

        # Calculate None statistics
        none_count = rfam_counts.get(None, 0)
        total_count = sum(rfam_counts.values())
        none_percentage = (none_count / total_count * 100) if total_count > 0 else 0

        print("\nRFAM family counts:")
        print("==================")

        # Sort by count (descending) for better readability
        for family, count in rfam_counts.most_common():
            if family is None:
                print(f"NULL: {count}")
            else:
                print(f"{family}: {count}")

        print(f"\nTotal families (including NULL): {len(rfam_counts)}")
        print(f"Total elements: {total_count}")
        print(f"NULL/None values: {none_count} ({none_percentage:.2f}%)")
        
        return rfam_counts
        
    except Exception as e:
        print(f"Error analyzing task {task_name}: {str(e)}")
        return Counter()


def main():
    # Set up logging to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = f"task_analysis_furna_stdout_{timestamp}.log"
    stderr_log = f"task_analysis_furna_stderr_{timestamp}.log"
    
    # Create log directory if it doesn't exist (relative to current location)
    os.makedirs("../logs", exist_ok=True)
    stdout_log = os.path.join("../logs", stdout_log)
    stderr_log = os.path.join("../logs", stderr_log)
    
    # Open log files
    stdout_file = open(stdout_log, 'w', encoding='utf-8')
    stderr_file = open(stderr_log, 'w', encoding='utf-8')
    
    # Redirect stdout and stderr to log files while keeping console output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = TeeOutput(stdout_file, original_stdout)
    sys.stderr = TeeOutput(stderr_file, original_stderr)
    
    try:
        print("RFAM Family Analysis for All Tasks (using FURNA dataset)")
        print("=" * 60)
        print(f"Logging stdout to: {stdout_log}")
        print(f"Logging stderr to: {stderr_log}")
        
        # Load FURNA dataset mapping
        pdb_to_rfam = load_furna_mapping()
        
        # Dictionary to store results for each task
        all_task_results = {}
        
        # Analyze each task
        for task_name in TASKS:
            print(f"\n{'='*60}")
            print(f"ANALYZING TASK: {task_name}")
            print(f"{'='*60}")
            
            result = analyze_task(task_name, pdb_to_rfam)
            all_task_results[task_name] = result

        # Collect all unique RFAM families across all tasks
        all_families = set()
        for task_results in all_task_results.values():
            all_families.update(task_results.keys())
        
        # Sort families (NULL first, then alphabetically)
        sorted_families = sorted(all_families, key=lambda x: (x is not None, x))
        
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Total unique RFAM families across all tasks: {len(all_families)}")
        print(f"Tasks analyzed: {len(all_task_results)}")
        
        # Write results to CSV
        csv_filename = "rfam_family_counts_furna_raw.csv"
        print(f"\nWriting results to {csv_filename}...")
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['RFAM_Family'] + TASKS
            writer.writerow(header)
            
            # Write data for each RFAM family
            for family in sorted_families:
                family_name = "NULL" if family is None else family
                row = [family_name]
                
                # Add count for each task (0 if family not present in that task)
                for task_name in TASKS:
                    count = all_task_results[task_name].get(family, 0)
                    row.append(count)
                
                writer.writerow(row)
        
        print(f"Results successfully written to {csv_filename}")
        print(f"CSV structure: {len(sorted_families)} rows (RFAM families) × {len(TASKS)} columns (tasks)")
        
        # Print summary table with None statistics
        print(f"\nSummary table:")
        print("Task\t\tTotal Elements\tUnique Families\tNone Count\tNone %")
        print("-" * 80)
        for task_name in TASKS:
            total_elements = sum(all_task_results[task_name].values())
            unique_families = len(all_task_results[task_name])
            none_count = all_task_results[task_name].get(None, 0)
            none_percentage = (none_count / total_elements * 100) if total_elements > 0 else 0
            print(f"{task_name:<15}\t{total_elements:<10}\t{unique_families:<12}\t{none_count:<10}\t{none_percentage:.2f}%")
    
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Close log files
        stdout_file.close()
        stderr_file.close()
        
        print(f"\nLogging completed. Files saved:")
        print(f"  stdout: {stdout_log}")
        print(f"  stderr: {stderr_log}")


if __name__ == "__main__":
    main() 