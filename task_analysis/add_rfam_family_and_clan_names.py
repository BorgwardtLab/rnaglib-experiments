#!/usr/bin/env python3
"""
Complete script to add RFAM family descriptions and clan information using FTP data files.
"""

import csv
import requests
import time
import sys
import argparse
from typing import Optional, Tuple, Dict

def load_clan_data() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load clan membership and clan description data from the downloaded files.
    
    Returns:
        Tuple of (family_to_clan_mapping, clan_to_description_mapping)
    """
    # Load family to clan mapping
    family_to_clan = {}
    try:
        with open('data/clan_membership.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        clan_acc = parts[0]
                        family_acc = parts[1]
                        family_to_clan[family_acc] = clan_acc
    except FileNotFoundError:
        print("Warning: data/clan_membership.txt not found. Clan accessions will be empty.")
    
    # Load clan descriptions
    clan_to_description = {}
    try:
        with open('data/clan.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        clan_acc = parts[0]
                        clan_description = parts[3]  # Short description
                        clan_to_description[clan_acc] = clan_description
    except FileNotFoundError:
        print("Warning: data/clan.txt not found. Clan descriptions will be empty.")
    
    print(f"Loaded {len(family_to_clan)} family-to-clan mappings")
    print(f"Loaded {len(clan_to_description)} clan descriptions")
    
    return family_to_clan, clan_to_description

def get_rfam_family_description(rfam_accession: str) -> Optional[str]:
    """
    Fetch the description for an RFAM family using the RFAM API.
    
    Args:
        rfam_accession: RFAM accession number (e.g., 'RF00001')
        
    Returns:
        Description string or None if not found
    """
    if rfam_accession == 'NULL' or not rfam_accession.startswith('RF'):
        return None
        
    url = f"https://rfam.org/family/{rfam_accession}?content-type=application/json"
    
    try:
        print(f"Fetching description for {rfam_accession}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            rfam_data = data.get('rfam', {})
            description = rfam_data.get('description', 'Description not available').strip()
            return description
        else:
            print(f"  Error: HTTP {response.status_code} for {rfam_accession}")
            return "Error fetching description"
            
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {rfam_accession}: {e}")
        return "Error fetching description"
    except Exception as e:
        print(f"  Unexpected error for {rfam_accession}: {e}")
        return "Error fetching description"

def main():
    parser = argparse.ArgumentParser(
        description="Add RFAM family descriptions and clan information to CSV file"
    )
    parser.add_argument(
        "--input", 
        "-i",
        default="rfam_family_counts_combined_raw.csv",
        help="Input CSV file (default: rfam_family_counts_combined_raw.csv)"
    )
    parser.add_argument(
        "--output", 
        "-o",
        default="rfam_family_counts_combined.csv",
        help="Output CSV file (default: rfam_family_counts_combined.csv)"
    )
    
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    
    print("Loading clan data from FTP files...")
    family_to_clan, clan_to_description = load_clan_data()
    
    print(f"\nReading {input_file}...")
    
    # Read the CSV file
    rows = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    print(f"Found {len(rows)-1} data rows")
    
    # Modify the header to insert new columns after the first column (RFAM_Family)
    new_header = [rows[0][0]]  # RFAM_Family
    new_header.extend(['Description', 'Clan_Accession', 'Clan_Description'])  # New columns
    new_header.extend(rows[0][1:])  # Rest of original columns
    rows[0] = new_header
    
    # Process each data row
    for i, row in enumerate(rows[1:], 1):
        rfam_accession = row[0]
        
        if rfam_accession == 'NULL':
            description = "No RFAM family"
            clan_accession = ""
            clan_description = ""
        else:
            # Get description from API
            description = get_rfam_family_description(rfam_accession)
            if description is None:
                description = "Description not available"
            
            # Get clan information from loaded data
            clan_accession = family_to_clan.get(rfam_accession, "")
            clan_description = ""
            if clan_accession:
                clan_description = clan_to_description.get(clan_accession, "")
        
        # Insert new data after the first column
        new_row = [row[0]]  # RFAM_Family
        new_row.extend([description, clan_accession, clan_description])  # New columns
        new_row.extend(row[1:])  # Rest of original columns
        rows[i] = new_row
        
        # Add a small delay to be respectful to the API
        if rfam_accession != 'NULL':
            time.sleep(0.5)
        
        clan_info = f" | Clan: {clan_accession} ({clan_description[:30]}{'...' if len(clan_description) > 30 else ''})" if clan_accession else " | No clan"
        print(f"Processed {i}/{len(rows)-1}: {rfam_accession} -> {description[:30]}{'...' if len(description) > 30 else ''}{clan_info}")
    
    # Write the updated CSV
    print(f"\nWriting complete CSV to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Done! Complete CSV saved as {output_file}")
    
    # Print summary statistics
    families_with_clans = sum(1 for row in rows[1:] if row[2])  # Count non-empty clan accessions
    print(f"\nSummary:")
    print(f"- Total families processed: {len(rows)-1}")
    print(f"- Families with clan assignments: {families_with_clans}")
    print(f"- Families without clan assignments: {len(rows)-1-families_with_clans}")

if __name__ == "__main__":
    main() 