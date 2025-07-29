#!/usr/bin/env python3
"""
NAKB functional annotation analyser script that extracts functional annotation information 
from RNA data for all available tasks using the NAKB database.
"""

import csv
import sys
import os
import json
import re
from datetime import datetime
from collections import Counter, defaultdict
from rnaglib.tasks import get_task

# Add parent directory to path to import our NAKB functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from find_functional_annotations import find_functional_annotations, FUNCTIONAL_TERMS, extract_hierarchy_from_js

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


def load_nakb_mapping():
    """
    Load NAKB database and create a mapping from PDB_ID to functional annotations.
    
    Returns:
        dict: Dictionary mapping PDB IDs to functional annotations
    """
    print("Loading NAKB database mapping...")
    nakb_path = "data/nakb.json"
    
    if not os.path.exists(nakb_path):
        print(f"Warning: NAKB database not found at {nakb_path}")
        return {}
    
    pdb_to_functional = {}
    
    try:
        with open(nakb_path, 'r') as f:
            data = json.load(f)
        
        print(f"NAKB database loaded with {data['response']['numFound']} entries")
        
        for doc in data['response']['docs']:
            pdb_id = doc.get('pdbid', '').lower()
            annotations = doc.get('NAKBna.entityannot', [])
            
            if pdb_id:
                # Extract functional annotations only
                functional_annotations = []
                for annot_string in annotations:
                    terms = [term.strip() for term in annot_string.split(',')]
                    functional_terms = [term for term in terms if term in FUNCTIONAL_TERMS]
                    functional_annotations.extend(functional_terms)
                
                # Remove duplicates while preserving order
                unique_functional = list(dict.fromkeys(functional_annotations))
                
                if unique_functional:  # Only store if there are functional annotations
                    pdb_to_functional[pdb_id] = unique_functional
                    
    except Exception as e:
        print(f"Error loading NAKB database: {e}")
        return {}
    
    print(f"Loaded {len(pdb_to_functional)} PDB-functional annotation mappings from NAKB database")
    return pdb_to_functional


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


def analyze_task(task_name, pdb_to_functional):
    """
    Analyze a single task and return functional annotation counts.
    
    Args:
        task_name (str): Name of the task to analyze
        pdb_to_functional (dict): Mapping from PDB IDs to functional annotations
        
    Returns:
        tuple: (functional_counts, task_description) - Counter object with functional annotation counts and task description dict
    """
    print(f"\n{'='*50}")
    print(f"Analyzing task: {task_name}")
    print(f"{'='*50}")
    
    try:
        # Load the task with specific root directory
        root_dir = f"./analyser_roots/{task_name}"
        print(f"Using root directory: {root_dir}")
        ta = get_task(task_id=task_name, root=root_dir, precomputed=True)
    
        # Get task description
        task_info = ta.describe()
        print(f"Task description: {task_info}")
        
        # Flatten the task description for easier CSV handling
        flattened_task_info = flatten_dict(task_info)
        
        print("\nExtracting functional annotations from NAKB database...")
        
        # Extract functional annotations
        functional_annotation_list = []
        found_count = 0
        not_found_count = 0
        
        for i in range(len(ta.dataset)):
            rna_data = ta.dataset[i]
            rna_graph = rna_data['rna']
            rna_name = rna_graph.name
            
            # Extract PDB ID from RNA name
            pdb_id = extract_pdb_id(rna_name)
            
            # Look up functional annotations in NAKB
            if pdb_id in pdb_to_functional:
                functional_annotations = pdb_to_functional[pdb_id]
                found_count += 1
                
                # Add each functional annotation separately to count them individually
                for annotation in functional_annotations:
                    functional_annotation_list.append(annotation)
                    
                # Also add a combined entry for structures with multiple annotations
                if len(functional_annotations) == 1:
                    # Single annotation - already added above
                    pass
                else:
                    # Multiple annotations - create a combined entry
                    combined = "+".join(sorted(functional_annotations))
                    functional_annotation_list.append(f"MULTI:{combined}")
                    
            else:
                functional_annotation_list.append(None)
                not_found_count += 1

        print(f"Number of elements: {len(ta.dataset)}")
        print(f"Found functional annotations: {found_count}")
        print(f"Not found (None): {not_found_count}")
        print(f"Total functional annotation entries: {len(functional_annotation_list)}")
        print(f"First 10 functional annotations: {functional_annotation_list[:10]}")
        print(f"Unique functional annotations: {set(functional_annotation_list)}")
        print(f"Number of unique annotations: {len(set(functional_annotation_list))}")
        
        # Count occurrences of each functional annotation type (including nulls)
        functional_counts = Counter(functional_annotation_list)

        # Calculate None statistics
        none_count = functional_counts.get(None, 0)
        total_count = sum(functional_counts.values())
        none_percentage = (none_count / total_count * 100) if total_count > 0 else 0

        print("\nFunctional annotation counts:")
        print("============================")

        # Sort by count (descending) for better readability
        for annotation, count in functional_counts.most_common():
            if annotation is None:
                print(f"NULL: {count}")
            else:
                print(f"{annotation}: {count}")

        print(f"\nTotal annotations (including NULL): {len(functional_counts)}")
        print(f"Total elements: {total_count}")
        print(f"NULL/None values: {none_count} ({none_percentage:.2f}%)")
        
        return functional_counts, flattened_task_info
        
    except Exception as e:
        print(f"Error analyzing task {task_name}: {str(e)}")
        return Counter(), {}


def extract_hierarchy_from_js_local():
    """Extract hierarchy information from NAKBnadict.js file (local version)."""
    
    with open('data/NAKBnadict.js', 'r') as f:
        content = f.read()
    
    entries = {}
    pattern = r'"([^"]+)":\s*{\s*"name":\s*"([^"]+)",\s*"hierarchy":\s*"([^"]+)",\s*"descr":\s*"([^"]*)"'
    matches = re.findall(pattern, content)
    
    for key, name, hierarchy, description in matches:
        entries[key] = {
            'name': name,
            'hierarchy': hierarchy,
            'description': description
        }
    
    return entries


def create_hierarchical_summary(all_task_results):
    """
    Create a hierarchical summary of functional annotations across all tasks.
    
    Args:
        all_task_results (dict): Dictionary of task results
        
    Returns:
        dict: Hierarchical summary of annotations
    """
    print("\nCreating hierarchical summary...")
    
    # Load hierarchy data
    try:
        hierarchy_data = extract_hierarchy_from_js_local()
    except Exception as e:
        print(f"Warning: Could not load hierarchy data: {e}")
        return {}
    
    # Aggregate counts by hierarchy level
    hierarchy_summary = defaultdict(lambda: defaultdict(int))
    
    for task_name, results in all_task_results.items():
        for annotation, count in results.items():
            if annotation is None or annotation.startswith('MULTI:'):
                continue
                
            if annotation in hierarchy_data:
                hierarchy_path = hierarchy_data[annotation]['hierarchy']
                levels = hierarchy_path.split(' > ')
                
                # Count at each level
                for i, level in enumerate(levels):
                    level_key = f"L{i}_{level}"
                    hierarchy_summary[level_key][task_name] += count
    
    return hierarchy_summary


def main():
    # Set up logging to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = f"task_analysis_nakb_stdout_{timestamp}.log"
    stderr_log = f"task_analysis_nakb_stderr_{timestamp}.log"
    
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
        print("NAKB Functional Annotation Analysis for All Tasks")
        print("=" * 60)
        print("Using NAKB database for functional annotation lookup")
        print(f"Logging stdout to: {stdout_log}")
        print(f"Logging stderr to: {stderr_log}")
        
        # Load NAKB database mapping
        pdb_to_functional = load_nakb_mapping()
        
        # Dictionary to store results for each task
        all_task_results = {}
        all_task_descriptions = {}
        
        # Analyze each task
        for task_name in TASKS:
            print(f"\n{'='*60}")
            print(f"ANALYZING TASK: {task_name}")
            print(f"{'='*60}")
            
            result, task_description = analyze_task(task_name, pdb_to_functional)
            all_task_results[task_name] = result
            all_task_descriptions[task_name] = task_description

        # Collect all unique functional annotations across all tasks
        all_annotations = set()
        for task_results in all_task_results.values():
            all_annotations.update(task_results.keys())
        
        # Sort annotations (NULL first, then alphabetically)
        sorted_annotations = sorted(all_annotations, key=lambda x: (x is not None, x))
        
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Total unique functional annotations across all tasks: {len(all_annotations)}")
        print(f"Tasks analyzed: {len(all_task_results)}")
        
        # Write functional annotation results to CSV
        csv_filename = "nakb_functional_annotation_counts.csv"
        print(f"\nWriting functional annotation results to {csv_filename}...")
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Functional_Annotation'] + TASKS
            writer.writerow(header)
            
            # Write data for each functional annotation
            for annotation in sorted_annotations:
                annotation_name = "NULL" if annotation is None else annotation
                row = [annotation_name]
                
                # Add count for each task (0 if annotation not present in that task)
                for task_name in TASKS:
                    count = all_task_results[task_name].get(annotation, 0)
                    row.append(count)
                
                writer.writerow(row)
        
        print(f"Functional annotation results successfully written to {csv_filename}")
        print(f"CSV structure: {len(sorted_annotations)} rows (functional annotations) × {len(TASKS)} columns (tasks)")
        
        # Write task descriptions to CSV
        task_desc_filename = "nakb_task_descriptions.csv"
        print(f"\nWriting task descriptions to {task_desc_filename}...")
        
        # Collect all unique description fields across all tasks
        all_desc_fields = set()
        for task_desc in all_task_descriptions.values():
            all_desc_fields.update(task_desc.keys())
        
        # Sort fields alphabetically for consistent column order
        sorted_desc_fields = sorted(all_desc_fields)
        
        with open(task_desc_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header (Task name + all description fields)
            header = ['Task_Name'] + sorted_desc_fields
            writer.writerow(header)
            
            # Write data for each task
            for task_name in TASKS:
                row = [task_name]
                task_desc = all_task_descriptions[task_name]
                
                # Add value for each description field (empty string if not present)
                for field in sorted_desc_fields:
                    value = task_desc.get(field, '')
                    row.append(value)
                
                writer.writerow(row)
        
        print(f"Task descriptions successfully written to {task_desc_filename}")
        print(f"Task descriptions CSV structure: {len(TASKS)} rows (tasks) × {len(sorted_desc_fields)} columns (description fields)")
        
        # Create hierarchical summary
        hierarchy_summary = create_hierarchical_summary(all_task_results)
        if hierarchy_summary:
            hierarchy_filename = "nakb_hierarchical_summary.csv"
            print(f"\nWriting hierarchical summary to {hierarchy_filename}...")
            
            with open(hierarchy_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = ['Hierarchy_Level'] + TASKS
                writer.writerow(header)
                
                # Write data for each hierarchy level
                for level_key in sorted(hierarchy_summary.keys()):
                    row = [level_key]
                    for task_name in TASKS:
                        count = hierarchy_summary[level_key].get(task_name, 0)
                        row.append(count)
                    writer.writerow(row)
            
            print(f"Hierarchical summary successfully written to {hierarchy_filename}")
        
        # Print summary table with None statistics
        print(f"\nSummary table:")
        print("Task\t\tTotal Elements\tUnique Annotations\tNone Count\tNone %")
        print("-" * 85)
        for task_name in TASKS:
            total_elements = sum(all_task_results[task_name].values())
            unique_annotations = len(all_task_results[task_name])
            none_count = all_task_results[task_name].get(None, 0)
            none_percentage = (none_count / total_elements * 100) if total_elements > 0 else 0
            print(f"{task_name:<15}\t{total_elements:<10}\t{unique_annotations:<12}\t{none_count:<10}\t{none_percentage:.2f}%")
    
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
        print(f"\nOutput files created:")
        print(f"  - nakb_functional_annotation_counts.csv")
        print(f"  - nakb_task_descriptions.csv")
        print(f"  - nakb_hierarchical_summary.csv")


if __name__ == "__main__":
    main() 