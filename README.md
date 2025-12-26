# CSC 590 – Final Project: Modeling Complex Binary-Class Associations in Simulated Genomic Data

**Author:** Mantra Mehta  
**Course:** CSC 590 – Graduate Project  
**University:** California State University, Dominguez Hills  
**Semester:** Fall 2025

## Project Overview

This project evaluates how different machine learning models perform when predicting binary genetic outcomes using simulated SNP datasets. Four genetic architectures were studied:

- 4-way Additive
- 2-way Epistatic
- 2-Additive + 2-way Epistatic Hybrid
- 4-way Heterogeneous

Each architecture was tested under:

- Low-dimensional datasets (100 SNPs)
- High-dimensional datasets (10,000 SNPs with missing values)

The models evaluated were:

- Logistic Regression
- Random Forest
- XGBoost

A filter-based Mutual Information feature selection method was applied to high-dimensional datasets to reduce dimensionality.

## Repository Structure

```
CSC590_FinalProject_MantraMehta/
│
├── Final-report/
│   ├── Mehta_Mantra_CSC590_Final-Report.pdf
│   ├── Mehta_Mantra_CSC590_Final-Report.dotx
│   └── Figures/
│
├── Code/
│   ├── report2_run.py          # Baseline models (no MI)
│   ├── test_run_r1.py          # Used for Report 1 experiments
│   ├── test_run_r2.py          # Used for Report 2 experiments
│   ├── report3_run_mi.py       # Final MI-based high-dimensional pipeline
│   ├── test_plot_r1.py         # ROC plotting utilities
│   └── test_plot_r2.py
│
├── Data/
│   ├── 2-wayEpi_100feat.txt
│   ├── 4-wayAdditive_100feat.txt
│   ├── 2-wayEpi_10000feat_with_NA.txt
│   ├── 2Additive_2-wayEpi_100feat.txt
│   ├── 2Additive_2-wayEpi_10000feat_with_NA.txt
│   ├── 4-wayAdditive_10000feat_with_NA.txt
│   ├── 4-wayHeterogeneous_100feat.txt
│   └── 4-wayHeterogeneous_10000feat_with_NA.txt
│
├── Results_CSV/
│   ├── metrics_*.csv
│   ├── mi_scores_*.csv
│   └── README_results.md (optional)
│
└── Report1_Report2_Report3/
    ├── Report_1_Project_Progress.pdf
    ├── Report_2_Project_Progress.pdf
    └── Report_3_Project_Progress.pdf
```

## Running the Code

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run the low-dimensional models (Report 2)

```bash
python Code/report2_run.py
```

### Run the high-dimensional MI pipeline (Report 3)

```bash
python Code/report3_run_mi.py
```

Each script automatically:

- Loads the dataset
- Applies preprocessing
- Runs Logistic Regression, Random Forest, and XGBoost
- Generates ROC curves, AUC metrics, and feature importance plots
- Saves outputs into the `Results_CSV/` and `Final-report/Figures/` directories

## Outputs

The scripts generate:

- ROC Curves for each dataset & model
- AUC metrics CSV files
- Feature importance plots (RF & XGBoost)
- Mutual Information score CSVs (high-dimensional only)

All figures used in the Final Report are located in:

`Final-report/Figures/`

## Citation

If referencing the pipeline or report, cite:

Mehta, M. (2025). Modeling Complex Binary-Class Associations in Simulated Genomic Data. CSC 590 Graduate Project, California State University, Dominguez Hills.

## Contact

For questions or academic verification:

- **Mantra Mehta**
- MS in Computer Science
- CSUDH — 2025
- GitHub: [@mantramehta](https://github.com/mantramehta)

