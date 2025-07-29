#!/usr/bin/env python3
"""
Script to create pie charts for family and clan distributions in each task directory.
Creates 4 charts per task: families (with/without NULL) and clans (with/without NULL).
"""

import csv
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


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
    plt.figure(figsize=(10, 8))
    
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


def process_task_directory(task_name):
    """
    Process a single task directory and create pie charts.
    
    Args:
        task_name: Name of the task directory
    """
    task_dir = task_name
    csv_file = os.path.join(task_dir, "family_clan_distribution.csv")
    
    if not os.path.exists(csv_file):
        print(f"Skipping {task_name} - no CSV file found")
        return
    
    print(f"\nProcessing {task_name}...")
    
    # Read the CSV data
    family_counts = Counter()
    clan_counts = Counter()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            family = row['RFAM_Family']
            clan = row['Clan_Accession']
            count = int(row['Count'])
            
            # Count families (use description if available, otherwise accession)
            family_desc = row['Description']
            family_label = f"{family} ({family_desc})" if family_desc else family
            family_counts[family_label] += count
            
            # Count clans (use description if available, otherwise accession)
            if clan:
                clan_desc = row['Clan_Description']
                clan_label = f"{clan} ({clan_desc})" if clan_desc else clan
                clan_counts[clan_label] += count
            else:
                clan_counts['No clan'] += count
    
    # Create family distributions
    # 1. Family distribution including NULL
    family_with_null = group_small_categories(dict(family_counts))
    create_pie_chart(family_with_null, 
                    f"{task_name.upper()}: Family Distribution (including NULL)",
                    "family_distribution_with_null.png", task_dir)
    
    # 2. Family distribution excluding NULL
    family_without_null = {k: v for k, v in family_counts.items() if not k.startswith('NULL')}
    if family_without_null:
        family_without_null = group_small_categories(family_without_null)
        create_pie_chart(family_without_null,
                        f"{task_name.upper()}: Family Distribution (excluding NULL)",
                        "family_distribution_without_null.png", task_dir)
    else:
        print(f"  No non-NULL families for {task_name}")
    
    # Create clan distributions
    # 3. Clan distribution including 'No clan'
    clan_with_none = group_small_categories(dict(clan_counts))
    create_pie_chart(clan_with_none,
                    f"{task_name.upper()}: Clan Distribution (including No clan)",
                    "clan_distribution_with_none.png", task_dir)
    
    # 4. Clan distribution excluding 'No clan'
    clan_without_none = {k: v for k, v in clan_counts.items() if k != 'No clan'}
    if clan_without_none:
        clan_without_none = group_small_categories(clan_without_none)
        create_pie_chart(clan_without_none,
                        f"{task_name.upper()}: Clan Distribution (excluding No clan)",
                        "clan_distribution_without_none.png", task_dir)
    else:
        print(f"  No clans assigned for {task_name}")


def main():
    """
    Main function to process all task directories.
    """
    # Task directories to process
    task_names = [
        "rna_cm", "rna_go", "rna_if", "rna_if_bench",
        "rna_ligand", "rna_prot", "rna_site", "rna_site_bench"
    ]
    
    print("Creating pie charts for task distributions...")
    print("=" * 50)
    
    # Process each task directory
    for task_name in task_names:
        if os.path.exists(task_name):
            process_task_directory(task_name)
        else:
            print(f"Directory {task_name} not found - skipping")
    
    print(f"\n{'=' * 50}")
    print("PIE CHARTS CREATED SUCCESSFULLY")
    print(f"{'=' * 50}")
    
    # Summary
    total_charts = 0
    for task_name in task_names:
        if os.path.exists(task_name):
            chart_files = [f for f in os.listdir(task_name) if f.endswith('.png')]
            print(f"✅ {task_name}: {len(chart_files)} charts")
            total_charts += len(chart_files)
    
    print(f"\nTotal charts created: {total_charts}")
    print("\nChart types per task:")
    print("  - family_distribution_with_null.png")
    print("  - family_distribution_without_null.png") 
    print("  - clan_distribution_with_none.png")
    print("  - clan_distribution_without_none.png")


if __name__ == "__main__":
    main() 