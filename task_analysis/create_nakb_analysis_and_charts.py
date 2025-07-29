#!/usr/bin/env python3
"""
Combined script to create individual task directories with functional annotation distribution tables
and generate pie charts for NAKB functional annotations.

Treats functional annotations as equivalent to RFAM families and L1 hierarchy categories as equivalent to RFAM clans.
"""

import csv
import os
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


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


def get_l1_category_mapping():
    """
    Create a mapping from functional annotations to their L1 categories.
    
    Returns:
        dict: Mapping of annotation -> L1 category info
    """
    try:
        hierarchy_data = extract_hierarchy_from_js_local()
    except Exception as e:
        print(f"Warning: Could not load hierarchy data: {e}")
        return {}
    
    annotation_to_l1 = {}
    
    for annotation, info in hierarchy_data.items():
        hierarchy_path = info['hierarchy'].split(' > ')
        
        if hierarchy_path[0] == 'function' and len(hierarchy_path) >= 2:
            l1_category = hierarchy_path[1]
            
            # Find the L1 category info
            l1_info = hierarchy_data.get(l1_category, {})
            
            annotation_to_l1[annotation] = {
                'l1_key': l1_category,
                'l1_name': l1_info.get('name', l1_category),
                'l1_description': l1_info.get('description', '')
            }
    
    return annotation_to_l1


def group_small_categories(data_dict, min_percentage=3.0):
    """
    Group categories with less than min_percentage into 'Others'.
    
    Args:
        data_dict: Dictionary of category -> count
        min_percentage: Minimum percentage to keep as separate category
        
    Returns:
        Dictionary with small categories grouped as 'Others'
    """
    total = sum(data_dict.values())
    if total == 0:
        return data_dict
    
    grouped = {}
    others_count = 0
    
    for category, count in data_dict.items():
        percentage = (count / total) * 100
        if percentage >= min_percentage:
            grouped[category] = count
        else:
            others_count += count
    
    if others_count > 0:
        grouped['Others'] = others_count
    
    return grouped


def create_pie_chart(data_dict, title, filename, task_dir):
    """
    Create a pie chart from data dictionary.
    
    Args:
        data_dict: Dictionary of category -> count
        title: Title for the pie chart
        filename: Filename to save the chart
        task_dir: Directory to save the chart in
    """
    if not data_dict or sum(data_dict.values()) == 0:
        print(f"  No data for {title} - skipping")
        return
    
    # Sort by count (descending)
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*sorted_items)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create pie chart
    colors = plt.cm.Set3.colors  # Use a colorful palette
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors)
    
    # Customize appearance
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    for text in texts:
        text.set_fontsize(8)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Save the chart
    chart_path = os.path.join(task_dir, filename)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created {chart_path}")


def create_task_directories_and_charts(input_file):
    """
    Read the NAKB functional annotation CSV and create individual task directories with
    distribution tables and pie charts.
    
    Args:
        input_file: Path to the input CSV file
    """
    
    # Task columns in the CSV
    task_columns = [
        "rna_cm", "rna_go", "rna_if", "rna_if_bench", 
        "rna_ligand", "rna_prot", "rna_site", "rna_site_bench"
    ]
    
    print(f"Reading {input_file}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    # Get L1 category mapping
    annotation_to_l1 = get_l1_category_mapping()
    print(f"Loaded hierarchy mapping for {len(annotation_to_l1)} functional annotations")
    
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Verify the structure
        print("CSV columns:", header)
        
        # Find task column indices
        task_indices = {}
        for i, col_name in enumerate(header):
            if col_name in task_columns:
                task_indices[col_name] = i
        
        print("Task column indices:", task_indices)
        
        # Store data for each task
        task_data = {task: [] for task in task_columns}
        
        # Process each row
        for row in reader:
            functional_annotation = row[0]
            
            # Skip MULTI entries for detailed analysis (but we'll count them in totals)
            if functional_annotation.startswith("MULTI:"):
                continue
            
            # For each task, if count > 0, add to that task's data
            for task_name in task_columns:
                if task_name in task_indices:
                    task_idx = task_indices[task_name]
                    count = int(row[task_idx]) if row[task_idx] else 0
                    
                    if count > 0:  # Only include annotations with non-zero counts
                        # Handle NULL entries
                        if functional_annotation == "NULL":
                            task_data[task_name].append({
                                'functional_annotation': 'NULL',
                                'l1_key': 'NULL',
                                'l1_name': 'No annotation',
                                'l1_description': 'No functional annotation found',
                                'count': count
                            })
                        else:
                            # Get L1 category info
                            l1_info = annotation_to_l1.get(functional_annotation, {})
                            
                            task_data[task_name].append({
                                'functional_annotation': functional_annotation,
                                'l1_key': l1_info.get('l1_key', 'Unknown'),
                                'l1_name': l1_info.get('l1_name', 'Unknown Category'),
                                'l1_description': l1_info.get('l1_description', ''),
                                'count': count
                            })
    
    # Create directories and files for each task
    for task_name in task_columns:
        print(f"\nProcessing task: {task_name}")
        
        # Create task directory
        task_dir = task_name
        os.makedirs(task_dir, exist_ok=True)
        
        # Sort by count (descending - most common to least common)
        sorted_data = sorted(task_data[task_name], key=lambda x: x['count'], reverse=True)
        
        # Calculate total count for this task
        total_count = sum(entry['count'] for entry in sorted_data)
        print(f"  Total functional annotations in {task_name}: {total_count}")
        
        if total_count == 0:
            print(f"  No functional annotations found for {task_name} - skipping")
            continue
        
        # ===============================
        # 1. Create functional annotation distribution CSV file
        # ===============================
        annotation_csv_filename = os.path.join(task_dir, "functional_annotation_distribution.csv")
        
        with open(annotation_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Functional_Annotation', 'L1_Category', 'L1_Description', 'Count', 'Percentage'])
            
            # Write data (sorted by count) with percentages
            for entry in sorted_data:
                percentage = (entry['count'] / total_count) * 100 if total_count > 0 else 0
                
                writer.writerow([
                    entry['functional_annotation'],
                    entry['l1_name'],
                    entry['l1_description'],
                    entry['count'],
                    f"{percentage:.2f}%"
                ])
        
        print(f"  Created {annotation_csv_filename} with {len(sorted_data)} functional annotations")
        
        # ===============================
        # 2. Aggregate data by L1 category
        # ===============================
        l1_data = defaultdict(lambda: {'count': 0, 'annotations': [], 'l1_description': ''})
        
        for entry in sorted_data:
            l1_key = entry['l1_key']
            l1_name = entry['l1_name']
            
            l1_data[l1_name]['count'] += entry['count']
            l1_data[l1_name]['annotations'].append(entry['functional_annotation'])
            if entry['l1_description']:
                l1_data[l1_name]['l1_description'] = entry['l1_description']
        
        # Sort L1 categories by count (descending)
        sorted_l1 = sorted(l1_data.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Create L1 category distribution CSV file
        l1_csv_filename = os.path.join(task_dir, "functional_category_distribution.csv")
        
        with open(l1_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['L1_Category', 'L1_Description', 'Count', 'Percentage', 'Number_of_Annotations'])
            
            # Write L1 category data with percentages
            for l1_name, l1_info in sorted_l1:
                l1_count = l1_info['count']
                percentage = (l1_count / total_count) * 100 if total_count > 0 else 0
                num_annotations = len(set(l1_info['annotations']))  # unique annotations
                
                writer.writerow([
                    l1_name,
                    l1_info['l1_description'],
                    l1_count,
                    f"{percentage:.2f}%",
                    num_annotations
                ])
        
        print(f"  Created {l1_csv_filename} with {len(sorted_l1)} L1 categories")
        
        # ===============================
        # 3. Create pie charts
        # ===============================
        print(f"  Creating pie charts for {task_name}...")
        
        # Prepare data for pie charts
        annotation_counts = {entry['functional_annotation']: entry['count'] for entry in sorted_data}
        l1_counts = {l1_name: l1_info['count'] for l1_name, l1_info in sorted_l1}
        
        # 3a. Functional annotation distribution INCLUDING NULL
        annotation_with_null = group_small_categories(annotation_counts)
        create_pie_chart(annotation_with_null, 
                        f"{task_name.upper()}: Functional Annotation Distribution (including NULL)",
                        "functional_annotation_distribution_with_null.png", task_dir)
        
        # 3b. Functional annotation distribution EXCLUDING NULL
        annotation_without_null = {k: v for k, v in annotation_counts.items() if k != 'NULL'}
        if annotation_without_null:
            annotation_without_null_grouped = group_small_categories(annotation_without_null)
            create_pie_chart(annotation_without_null_grouped,
                            f"{task_name.upper()}: Functional Annotation Distribution (excluding NULL)",
                            "functional_annotation_distribution_without_null.png", task_dir)
        else:
            print(f"  No non-NULL annotations for {task_name}")
        
        # 3c. L1 category distribution INCLUDING "No annotation"
        l1_with_none = group_small_categories(l1_counts)
        create_pie_chart(l1_with_none,
                        f"{task_name.upper()}: Functional Category Distribution (including No annotation)",
                        "functional_category_distribution_with_none.png", task_dir)
        
        # 3d. L1 category distribution EXCLUDING "No annotation"
        l1_without_none = {k: v for k, v in l1_counts.items() if k not in ['No annotation', 'Unknown Category']}
        if l1_without_none:
            l1_without_none_grouped = group_small_categories(l1_without_none)
            create_pie_chart(l1_without_none_grouped,
                            f"{task_name.upper()}: Functional Category Distribution (excluding No annotation)",
                            "functional_category_distribution_without_none.png", task_dir)
        else:
            print(f"  No annotated categories for {task_name}")
        
        # ===============================
        # 4. Show summaries
        # ===============================
        # Show top 5 functional annotations for this task
        if sorted_data:
            print(f"  Top 5 functional annotations in {task_name}:")
            for i, entry in enumerate(sorted_data[:5]):
                annotation = entry['functional_annotation']
                count = entry['count']
                percentage = (entry['count'] / total_count) * 100 if total_count > 0 else 0
                l1_name = entry['l1_name']
                print(f"    {i+1}. {annotation}: {count} ({percentage:.2f}%) - {l1_name}")
        
        # Show top 5 L1 categories for this task
        if sorted_l1:
            print(f"  Top 5 functional categories in {task_name}:")
            for i, (l1_name, l1_info) in enumerate(sorted_l1[:5]):
                l1_count = l1_info['count']
                percentage = (l1_count / total_count) * 100 if total_count > 0 else 0
                num_annotations = len(set(l1_info['annotations']))
                l1_desc = l1_info['l1_description'][:40] + "..." if len(l1_info['l1_description']) > 40 else l1_info['l1_description']
                print(f"    {i+1}. {l1_name}: {l1_count} ({percentage:.2f}%) - {num_annotations} annotations - {l1_desc}")
    
    print(f"\n{'='*80}")
    print("NAKB FUNCTIONAL ANNOTATION ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    # Final summary
    for task_name in task_columns:
        if task_data[task_name]:
            task_annotations = len(task_data[task_name])
            total_count = sum(entry['count'] for entry in task_data[task_name])
            
            # Count unique L1 categories for this task
            unique_l1 = set()
            for entry in task_data[task_name]:
                unique_l1.add(entry['l1_name'])
            
            print(f"✅ {task_name}/")
            print(f"   - functional_annotation_distribution.csv ({task_annotations} annotations, {total_count} total)")
            print(f"   - functional_category_distribution.csv ({len(unique_l1)} L1 categories)")
            
            # Count chart files
            if os.path.exists(task_name):
                chart_files = [f for f in os.listdir(task_name) if f.endswith('.png')]
                print(f"   - {len(chart_files)} pie charts")
    
    print(f"\nChart types created per task:")
    print("  - functional_annotation_distribution_with_null.png")
    print("  - functional_annotation_distribution_without_null.png")
    print("  - functional_category_distribution_with_none.png") 
    print("  - functional_category_distribution_without_none.png")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create task directories with functional annotation distributions and pie charts from NAKB CSV file."
    )
    
    parser.add_argument(
        'input_csv',
        nargs='?',  # Optional positional argument
        default='nakb_functional_annotation_counts.csv',
        help='Input CSV file containing functional annotation counts (default: nakb_functional_annotation_counts.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CREATING NAKB FUNCTIONAL ANNOTATION ANALYSIS AND CHARTS")
    print("=" * 80)
    print(f"Input file: {args.input_csv}")
    print("Functional annotations = RFAM families equivalent")
    print("L1 categories = RFAM clans equivalent")
    print("Creates 4 pie charts per task (with/without NULL values)")
    print()
    
    create_task_directories_and_charts(args.input_csv)


if __name__ == "__main__":
    main() 