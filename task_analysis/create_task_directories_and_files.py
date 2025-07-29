#!/usr/bin/env python3
"""
Script to create individual task directories with family/clan distribution tables
from the existing enhanced CSV file.
"""

import csv
import os
import argparse
from collections import defaultdict


def create_task_directories(input_file):
    """
    Read the enhanced CSV and create individual task directories with
    family/clan distribution tables sorted by frequency.
    
    Args:
        input_file: Path to the input CSV file
    """
    
    # Task columns in the CSV (positions 4-11)
    task_columns = [
        "rna_cm", "rna_go", "rna_if", "rna_if_bench", 
        "rna_ligand", "rna_prot", "rna_site", "rna_site_bench"
    ]
    
    print(f"Reading {input_file}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
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
            rfam_family = row[0]
            description = row[1]
            clan_accession = row[2]
            clan_description = row[3]
            
            # For each task, if count > 0, add to that task's data
            for task_name in task_columns:
                if task_name in task_indices:
                    task_idx = task_indices[task_name]
                    count = int(row[task_idx])
                    
                    if count > 0:  # Only include families with non-zero counts
                        task_data[task_name].append({
                            'rfam_family': rfam_family,
                            'description': description,
                            'clan_accession': clan_accession,
                            'clan_description': clan_description,
                            'count': count
                        })
    
    # Create directories and CSV files for each task
    for task_name in task_columns:
        print(f"\nProcessing task: {task_name}")
        
        # Create task directory
        task_dir = task_name
        os.makedirs(task_dir, exist_ok=True)
        
        # Sort by count (descending - most common to least common)
        sorted_data = sorted(task_data[task_name], key=lambda x: x['count'], reverse=True)
        
        # Calculate total count for this task
        total_count = sum(entry['count'] for entry in sorted_data)
        print(f"  Total RNAs in {task_name}: {total_count}")
        
        # Create family distribution CSV file
        csv_filename = os.path.join(task_dir, "family_clan_distribution.csv")
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header (including percentage column)
            writer.writerow(['RFAM_Family', 'Description', 'Clan_Accession', 'Clan_Description', 'Count', 'Percentage'])
            
            # Write data (sorted by count) with percentages
            for entry in sorted_data:
                # Calculate percentage
                percentage = (entry['count'] / total_count) * 100 if total_count > 0 else 0
                
                writer.writerow([
                    entry['rfam_family'],
                    entry['description'],
                    entry['clan_accession'],
                    entry['clan_description'],
                    entry['count'],
                    f"{percentage:.2f}%"
                ])
        
        print(f"  Created {csv_filename} with {len(sorted_data)} families")
        
        # Aggregate data by clan
        clan_data = defaultdict(lambda: {'count': 0, 'families': [], 'clan_description': ''})
        
        for entry in sorted_data:
            clan_acc = entry['clan_accession']
            if not clan_acc:  # Handle families without clans
                clan_acc = "No_Clan"
                clan_desc = "Families without clan assignment"
            else:
                clan_desc = entry['clan_description']
            
            clan_data[clan_acc]['count'] += entry['count']
            clan_data[clan_acc]['families'].append(entry['rfam_family'])
            if clan_desc:  # Keep the clan description
                clan_data[clan_acc]['clan_description'] = clan_desc
        
        # Sort clans by count (descending)
        sorted_clans = sorted(clan_data.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Create clan distribution CSV file
        clan_csv_filename = os.path.join(task_dir, "clan_distributions.csv")
        
        with open(clan_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Clan_Accession', 'Clan_Description', 'Count', 'Percentage', 'Number_of_Families'])
            
            # Write clan data with percentages
            for clan_acc, clan_info in sorted_clans:
                clan_count = clan_info['count']
                percentage = (clan_count / total_count) * 100 if total_count > 0 else 0
                num_families = len(clan_info['families'])
                
                writer.writerow([
                    clan_acc,
                    clan_info['clan_description'],
                    clan_count,
                    f"{percentage:.2f}%",
                    num_families
                ])
        
        print(f"  Created {clan_csv_filename} with {len(sorted_clans)} clans")
        
        # Show top 5 families for this task
        if sorted_data:
            print(f"  Top 5 families in {task_name}:")
            for i, entry in enumerate(sorted_data[:5]):
                family_name = entry['rfam_family']
                count = entry['count']
                percentage = (entry['count'] / total_count) * 100 if total_count > 0 else 0
                desc = entry['description'][:40] + "..." if len(entry['description']) > 40 else entry['description']
                print(f"    {i+1}. {family_name}: {count} ({percentage:.2f}%) - {desc}")
        
        # Show top 5 clans for this task
        if sorted_clans:
            print(f"  Top 5 clans in {task_name}:")
            for i, (clan_acc, clan_info) in enumerate(sorted_clans[:5]):
                clan_count = clan_info['count']
                percentage = (clan_count / total_count) * 100 if total_count > 0 else 0
                num_families = len(clan_info['families'])
                clan_desc = clan_info['clan_description'][:40] + "..." if len(clan_info['clan_description']) > 40 else clan_info['clan_description']
                print(f"    {i+1}. {clan_acc}: {clan_count} ({percentage:.2f}%) - {num_families} families - {clan_desc}")
    
    print(f"\n{'='*60}")
    print("TASK DIRECTORIES CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    
    for task_name in task_columns:
        task_families = len(task_data[task_name])
        total_rnas = sum(entry['count'] for entry in task_data[task_name])
        
        # Count unique clans for this task
        unique_clans = set()
        for entry in task_data[task_name]:
            clan_acc = entry['clan_accession'] if entry['clan_accession'] else "No_Clan"
            unique_clans.add(clan_acc)
        
        print(f"✅ {task_name}/")
        print(f"   - family_clan_distribution.csv ({task_families} families, {total_rnas} total RNAs)")
        print(f"   - clan_distributions.csv ({len(unique_clans)} clans)")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create individual task directories with family/clan distribution tables from enhanced CSV file."
    )
    
    parser.add_argument(
        'input_csv',
        nargs='?',  # Optional positional argument
        default='rfam_family_counts_combined.csv',
        help='Input CSV file containing family counts (default: rfam_family_counts_combined.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CREATING TASK DIRECTORIES AND FILES")
    print("=" * 60)
    print(f"Input file: {args.input_csv}")
    print()
    
    create_task_directories(args.input_csv)


if __name__ == "__main__":
    main() 