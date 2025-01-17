# Census_VAE

A Variational Auto-Encoder (VAE) model designed to predict disease labels based on single-cell sequencing data.

## Environment Setup

Ensure you are using the **[census_env](assets/requirements.txt)** environment to run the code.

```bash
conda create --name census_env --file assets/requirements.txt

conda activate census_env
```

## Preprocessing Steps

The following steps outline the correct order for preprocessing the data:

1. **[scripts/_make_gene_name_vocab.ipynb](scripts/_make_gene_name_vocab.ipynb)**  
   - After downloading the human gene file (refer to the notebook for the link), this script creates a vocabulary mapping gene names to their Ensembl IDs (useful for subsequent processing).

2. **[scripts/1_download_datasets.ipynb](scripts/1_download_datasets.ipynb)**  
   - Downloads, filters, and organizes datasets:
     - Filters for human data (`homo_sapiens`).
     - Ensures datasets contain non-zero unique cells.
     - Avoids duplicate datasets sharing the same cells.
     - Excludes cancer-related datasets based on predefined disease categories.
     - Handles missing titles and citations via manual mapping.
     - Saves metadata to a CSV file for later use.
     - Downloads datasets to a predefined folder.
   - Relevant resources:
     - [Census API Demo for Datasets](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_datasets.html)
     - [Census Query Extraction](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_query_extract.html)
   - Key Statistics:
     - Unique datasets: **290**
     - Unique collections: **112**
     - Unique cell types: **593**
     - Unique diseases: **40**

3. **[scripts/2_preprocess_datasets.py](scripts/2_preprocess_datasets.py)**  
   - Preprocessing large datasets requires significant resources; it is recommended to run this step on the EPFL cluster. Use the script [scripts/3_VAE_train.run](scripts/3_VAE_train.run) to launch this process.  
   - Main tasks performed:
     - Ensures datasets include required columns like `donor_id`, `cell_type`, and `disease`.
     - Applies normalization (check the script for the available normalization methods).
     - Merges all datasets into a single `AnnData` object, handling missing genes appropriately.
     - Concatenates and encodes labels (e.g., `cell_type_ontology_term_id-disease_ontology_term_id`) as integers.
     - Removes cells with duplicate barcodes.
     - Saves the processed dataset as `merged_30.h5ad`.


## Training

The VAE model supports parallelized training to save time. It can run on CPUs or GPUs, with specific support for [Data Parallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) GPU training. A summary of supported configurations:

- **CPU**: Not recommended (too slow).
- **GPU DDP**: Code exists but is not functional due to the complexity of parallelizing multiple GPU processes.
- **GPU DP (Recommended)**:
  - Model Architecture: [model/VAE_parallel_gpu_dp.py](model/VAE_parallel_gpu_dp.py)
  - Training Code: [model/train_parallel_gpu_dp.py](model/train_parallel_gpu_dp.py)
  - SCITAS Run Script: [scripts/3_VAE_train_gpu.run](scripts/3_VAE_train_gpu.run)

### Important Notes
- **Hyperparameters:** Before launching training, verify and update the hyperparameters (e.g., number of epochs) in the scripts.
- **Dataset Splitting:** 
  - The dataset is split into training and test sets in a consistent manner.
  - The training set is further divided into training and validation subsets based on the variable `SEED_CROSSVALIDATION`. 
  - By running the script multiple times, you can perform cross-validation.
- **Running time:**
    - All data (30 datasets) | batchsize=512 | 2 CHF per epoch | 1h per epoch

