# RNA Task Analysis: RFAM Families and NAKB Functional Annotations

This directory contains complete workflows for analyzing both RFAM family distributions and NAKB functional annotation distributions across RNA tasks, with comprehensive visualizations and per-task breakdowns.

## Overview

This analysis suite provides two complementary approaches for understanding RNA task compositions:

### RFAM Family Analysis
The RFAM pipeline processes RNA task data to extract RFAM family information using multiple approaches, combines the results with RFAM family descriptions and clan information, and generates comprehensive visualizations and per-task distributions.

### NAKB Functional Annotation Analysis
The NAKB pipeline extracts functional annotation information from the NAKB database, providing insights into the biological roles of RNA structures (protein synthesis, catalytic activity, regulation, etc.) with hierarchical categorization and visualizations.

## Scripts and Files

### Core Analysis Scripts

#### **`task_analyser.py`** 
- **Purpose**: Basic RFAM family extraction using RfamTransform
- **Input**: RNA task data via rnaglib
- **Output**: `rfam_family_counts_raw.csv`
- **Method**: Uses RfamTransform to extract RFAM families from task data

#### **`task_analyser_furna.py`**
- **Purpose**: RFAM family extraction using FURNA dataset lookup
- **Input**: RNA task data + FURNA dataset mapping
- **Output**: `rfam_family_counts_furna.csv`
- **Method**: Looks up PDB IDs in FURNA dataset to find RFAM families

#### **`task_analyser_combined.py`** ⭐ **RECOMMENDED**
- **Purpose**: Hierarchical approach combining both methods for maximum coverage
- **Input**: RNA task data + FURNA dataset mapping
- **Output**: `rfam_family_counts_combined_raw.csv`
- **Method**: 
  1. First tries FURNA dataset lookup by PDB ID
  2. If not found, tries RfamTransform
  3. Only returns None if both fail

#### **`add_rfam_family_and_clan_names.py`**
- **Purpose**: Enhances raw data with family descriptions and clan information
- **Input**: Raw CSV files + RFAM data files
- **Output**: Enhanced CSV files with descriptions and clan info
- **Features**:
  - Fetches RFAM family descriptions via API
  - Adds clan membership and descriptions from FTP data
  - Command-line interface with customizable inputs/outputs

### Visualization and Distribution Scripts

#### **`create_task_directories_and_files.py`**
- **Purpose**: Creates individual task directories with family/clan distribution tables
- **Input**: Enhanced CSV file (default: `rfam_family_counts_combined.csv`)
- **Output**: Individual directories for each task with `family_clan_distribution.csv`
- **Usage**: 
  ```bash
  python create_task_directories_and_files.py [input_csv]
  ```

#### **`create_pie_charts.py`**
- **Purpose**: Creates pie charts for family and clan distributions
- **Input**: Task directories with `family_clan_distribution.csv` files
- **Output**: 4 pie charts per task:
  - `family_distribution_with_null.png`
  - `family_distribution_without_null.png`
  - `clan_distribution_with_none.png`
  - `clan_distribution_without_none.png`
- **Features**:
  - Groups small categories (<3%) into "Others"
  - Shows families as "Accession (Description)"
  - Shows clans as "Accession (Description)"

### NAKB Functional Annotation Scripts

#### **`task_analyser_nakb.py`** ⭐ **NEW**
- **Purpose**: Extract functional annotations from NAKB database for all tasks
- **Input**: RNA task data + NAKB database (nakb.json)
- **Output**: 
  - `nakb_functional_annotation_counts.csv` - Main results matrix
  - `nakb_task_descriptions.csv` - Task metadata 
  - `nakb_hierarchical_summary.csv` - Hierarchical analysis
- **Method**: Maps PDB IDs to functional annotations (ribosomalrna, trna, nazyme, etc.)

#### **`create_nakb_analysis_and_charts.py`** ⭐ **NEW**
- **Purpose**: Creates task directories with functional annotation distributions and pie charts
- **Input**: `nakb_functional_annotation_counts.csv`
- **Output**: Individual task directories with:
  - `functional_annotation_distribution.csv` - Detailed annotations (like RFAM families)
  - `functional_category_distribution.csv` - L1 categories (like RFAM clans)
  - 4 pie charts per task (with/without NULL values)
- **Features**:
  - Two-level hierarchy: functional annotations → L1 categories
  - Groups small categories (<3%) into "Others"
  - Shows both detailed and categorical views

### Data Files

#### **RFAM Analysis Data Files**
- **`rfam_family_counts_combined.csv`** - ⭐ **MAIN RESULT** - Enhanced CSV with descriptions and clan information
- **`rfam_family_counts_combined_raw.csv`** - Raw counts from combined analysis approach
- **`rfam_family_counts.csv`** - Enhanced CSV from basic analysis
- **`rfam_family_counts_raw.csv`** - Raw counts from basic analysis  
- **`rfam_family_counts_furna.csv`** - Raw counts from FURNA-only analysis

#### **NAKB Analysis Data Files** ⭐ **NEW**
- **`nakb_functional_annotation_counts.csv`** - ⭐ **MAIN RESULT** - Functional annotation counts matrix
- **`nakb_task_descriptions.csv`** - Task metadata from NAKB analysis
- **`nakb_hierarchical_summary.csv`** - Hierarchical summary by L1 categories

#### **Task Directories**

**RFAM Analysis** (created by `create_task_directories_and_files.py`):
- **`rna_cm/`**, **`rna_go/`**, **`rna_if/`**, **`rna_if_bench/`**, **`rna_ligand/`**, **`rna_prot/`**, **`rna_site/`**, **`rna_site_bench/`**
  - Each contains `family_clan_distribution.csv` and 4 RFAM pie charts

**NAKB Analysis** (created by `create_nakb_analysis_and_charts.py`):
- **`rna_cm/`**, **`rna_go/`**, **`rna_if/`**, **`rna_if_bench/`**, **`rna_ligand/`**, **`rna_prot/`**, **`rna_site/`**, **`rna_site_bench/`**
  - Each contains `functional_annotation_distribution.csv`, `functional_category_distribution.csv`, and 4 functional annotation pie charts

#### **Compressed Task Data**
- **`*.tar.gz`** files - Compressed task data archives

#### **Support Data**
- **`data/`** directory - Contains RFAM clan membership and description files

## Complete Workflow

### Option A: Full Combined Analysis (Recommended)

```bash
# Step 1: Run combined analysis for maximum coverage
python3 task_analyser_combined.py

# Step 2: Enhance with descriptions and clan information
python3 add_rfam_family_and_clan_names.py rfam_family_counts_combined_raw.csv rfam_family_counts_combined.csv

# Step 3: Create individual task directories
python3 create_task_directories_and_files.py rfam_family_counts_combined.csv

# Step 4: Generate pie chart visualizations
python3 create_pie_charts.py
```

### Option B: Basic Analysis Only

```bash
# Step 1: Run basic analysis
python3 task_analyser.py

# Step 2: Enhance with descriptions and clan information  
python3 add_rfam_family_and_clan_names.py rfam_family_counts_raw.csv rfam_family_counts.csv

# Step 3: Create task directories and visualizations
python3 create_task_directories_and_files.py rfam_family_counts.csv
python3 create_pie_charts.py
```

### Option C: NAKB Functional Annotation Analysis ⭐ **NEW**

```bash
# Step 1: Run NAKB functional annotation analysis
python3 task_analyser_nakb.py

# Step 2: Create task directories and pie charts (combined)
python3 create_nakb_analysis_and_charts.py nakb_functional_annotation_counts.csv
```

## CSV Structure

### RFAM Analysis CSV Files

The enhanced RFAM CSV files have the following columns:

1. **`RFAM_Family`** - RFAM accession numbers (e.g., RF00001)
2. **`Description`** - Family descriptions from RFAM API  
3. **`Clan_Accession`** - Clan accession numbers (e.g., CL00012)
4. **`Clan_Description`** - Clan descriptions (e.g., "SAM clan")
5. **Task columns**: `rna_cm`, `rna_go`, `rna_if`, `rna_if_bench`, `rna_ligand`, `rna_prot`, `rna_site`, `rna_site_bench`

### NAKB Analysis CSV Files

The NAKB functional annotation CSV files have the following columns:

1. **`Functional_Annotation`** - Functional annotation terms (e.g., ribosomalrna, trna, nazyme)
2. **Task columns**: `rna_cm`, `rna_go`, `rna_if`, `rna_if_bench`, `rna_ligand`, `rna_prot`, `rna_site`, `rna_site_bench`

Per-task CSV files include hierarchical information:
- **`functional_annotation_distribution.csv`**: Functional_Annotation, L1_Category, L1_Description, Count, Percentage
- **`functional_category_distribution.csv`**: L1_Category, L1_Description, Count, Percentage, Number_of_Annotations

## Data Sources

### RFAM Analysis
- **Task analysis**: rnaglib package with RfamTransform and FURNA dataset
- **Family descriptions**: RFAM API (`https://rfam.org/family/{accession}?content-type=application/json`)
- **Clan mappings**: RFAM FTP (`https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/clan_membership.txt.gz`)
- **Clan descriptions**: RFAM FTP (`https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/clan.txt.gz`)

### NAKB Analysis  
- **Task analysis**: rnaglib package with NAKB database lookup
- **Functional annotations**: NAKB database (`data/nakb.json`) - 13,774 PDB structures with functional classifications
- **Hierarchy information**: NAKBnadict.js - functional annotation hierarchy and descriptions
- **PDB ID mapping**: Extracted from RNA task data (various formats: 1a1t, 1a1t.A, 1a1t_A)

## Output Structure

### After running RFAM workflow:

```
task_analysis/
├── rfam_family_counts_combined.csv          # Main enhanced RFAM results
├── rna_cm/
│   ├── family_clan_distribution.csv
│   ├── clan_distributions.csv
│   ├── family_distribution_with_null.png
│   ├── family_distribution_without_null.png
│   ├── clan_distribution_with_none.png
│   └── clan_distribution_without_none.png
├── rna_go/
│   └── [same structure]
└── [other task directories...]
```

### After running NAKB workflow:

```
task_analysis/
├── nakb_functional_annotation_counts.csv    # Main NAKB results
├── nakb_task_descriptions.csv
├── nakb_hierarchical_summary.csv
├── rna_cm/
│   ├── functional_annotation_distribution.csv
│   ├── functional_category_distribution.csv
│   ├── functional_annotation_distribution_with_null.png
│   ├── functional_annotation_distribution_without_null.png
│   ├── functional_category_distribution_with_none.png
│   └── functional_category_distribution_without_none.png
├── rna_go/
│   └── [same structure]
└── [other task directories...]
```

## Logging

- Analysis scripts create timestamped log files in `../logs/`
- Enhancement and visualization scripts output progress to console

## Statistics

### RFAM Analysis (Combined Analysis)
- **Total families**: 153
- **Families with clan assignments**: ~44%
- **Families without clan assignments**: ~56%
- **Tasks analyzed**: 8 (rna_cm, rna_go, rna_if, rna_if_bench, rna_ligand, rna_prot, rna_site, rna_site_bench)

### NAKB Analysis
- **Total PDB structures in database**: 13,774
- **Functional annotation terms**: 40+ (ribosomalrna, trna, nazyme, riboswitch, etc.)
- **L1 functional categories**: 11 (makesprotein, nazyme, riboswitch, transcription, etc.)
- **Hierarchical levels**: 3-4 levels (function > category > subcategory > specific)
- **Tasks analyzed**: 8 (rna_cm, rna_go, rna_if, rna_if_bench, rna_ligand, rna_prot, rna_site, rna_site_bench) 