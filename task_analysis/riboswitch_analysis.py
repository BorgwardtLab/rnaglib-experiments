#!/usr/bin/env python3
"""
Script to analyze riboswitch RNAs across multiple task directories.
Finds family_clan_distribution.csv files in each directory and calculates riboswitch statistics.
"""

import csv
import sys
import os

def analyze_riboswitch_data(csv_file_path):
    """
    Analyze the CSV file to find riboswitch entries and calculate statistics.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        tuple: (riboswitch_count, riboswitch_percentage, total_count, riboswitch_families)
    """
    riboswitch_count = 0
    total_count = 0
    riboswitch_families = set()
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                # Skip empty rows
                if not row.get('Description'):
                    continue
                    
                count = int(row['Count']) if row['Count'].isdigit() else 0
                total_count += count
                
                # Check if "riboswitch" is in the description (case-insensitive)
                if 'riboswitch' in row['Description'].lower():
                    riboswitch_count += count
                    riboswitch_families.add(row['RFAM_Family'])
    
    except FileNotFoundError:
        return None, None, None, None
    except Exception as e:
        print(f"Error reading file {csv_file_path}: {e}")
        return None, None, None, None
    
    # Calculate percentage of riboswitches
    riboswitch_percentage = (riboswitch_count / total_count * 100) if total_count > 0 else 0
    
    return riboswitch_count, riboswitch_percentage, total_count, len(riboswitch_families)

def main():
    # Directories to analyze
    directories = [
        "rna_cm",
        "rna_go", 
        "rna_if",
        "rna_if_bench",
        "rna_ligand",
        "rna_prot",
        "rna_site",
        "rna_site_bench"
    ]
    
    # Results storage
    results = []
    
    # Analyze each directory
    for directory in directories:
        csv_file = os.path.join(directory, "family_clan_distribution.csv")
        
        if os.path.exists(csv_file):
            riboswitch_count, riboswitch_percentage, total_count, riboswitch_families = analyze_riboswitch_data(csv_file)
            
            if riboswitch_count is not None:
                results.append({
                    'directory': directory,
                    'total_count': total_count,
                    'riboswitch_count': riboswitch_count,
                    'riboswitch_percentage': riboswitch_percentage,
                    'riboswitch_families': riboswitch_families
                })
            else:
                results.append({
                    'directory': directory,
                    'total_count': 'ERROR',
                    'riboswitch_count': 'ERROR',
                    'riboswitch_percentage': 'ERROR',
                    'riboswitch_families': 'ERROR'
                })
        else:
            results.append({
                'directory': directory,
                'total_count': 'N/A',
                'riboswitch_count': 'N/A',
                'riboswitch_percentage': 'N/A',
                'riboswitch_families': 'N/A'
            })
    
    # Print results table
    print("RIBOSWITCH ANALYSIS ACROSS TASK DIRECTORIES")
    print("=" * 80)
    print(f"{'Directory':<15} {'Total RNAs':<12} {'Riboswitches':<12} {'Percentage':<12} {'Families':<10}")
    print("-" * 80)
    
    for result in results:
        total_str = f"{result['total_count']:,}" if isinstance(result['total_count'], int) else str(result['total_count'])
        riboswitch_str = f"{result['riboswitch_count']:,}" if isinstance(result['riboswitch_count'], int) else str(result['riboswitch_count'])
        percentage_str = f"{result['riboswitch_percentage']:.2f}%" if isinstance(result['riboswitch_percentage'], float) else str(result['riboswitch_percentage'])
        families_str = str(result['riboswitch_families'])
        
        print(f"{result['directory']:<15} {total_str:<12} {riboswitch_str:<12} {percentage_str:<12} {families_str:<10}")

if __name__ == "__main__":
    main()