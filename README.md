# census_VAE

Variational Auto-Encoder model thta tries to predict label diasease based on single cell data sequencing.

## Environment Setup
The code is designed to run in the **[census_env](assets/requirements.txt)** environment.

## Preprocessing

Here are reported in the corredt podert the files to preprocess the data. 

1) **[scripts/_make_gene_name_vocab.ipynb](scripts/_make_gene_name_vocab.ipynb)**: After downloading the human gene file (please refoer tot the notebook for the link), it create a vocaboltu thta maps eahc gene names ot its ensebale ID (useful for later processing).

2) **[scripts/1_download_datasets.ipynb](scripts/1_download_datasets.ipynb)**: process for downloading, filtering, and organizing datasets:
   - Filters datasets based on several criteria:
     - Selects datasets containing human data (`homo_sapiens`).
     - Datasets with non-zero unique cells.
        - **ATTENTION**: multiple datasets can share the same cells, thus these duplicates MUST be avoided.
     - Exclusion of datasets with duplicate entries.
     - Removal of `cancer-related` datasets based on predefined disease categories.
   - Handles missing titles and citations by manual mapping.
   - Saves metadata to a CSV file for later use.
   - Download datasets, stored in a predefined folder.
   - P.S. Census data management done using: [Census API Demo for Datasets](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_datasets.html), [Census Query Extraction](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_query_extract.html).
   - Useful:
        - Number of unique datasets: 290
        - Number of unique collections: 112
        - Number of unique cell types: 593
        - Number of unique diseases: 40

3) **[scripts/2_preprocess_datasets.py](scripts/2_preprocess_datasets.py)**: As processing all the dataset can be long and requiring lot of memory, to run this step you need to use the EPFL cluster, thus lunch this script using [scripts/3_VAE_train.run](scripts/3_VAE_train.run). The script perform:
   - Ensures that each dataset includes required columns like `donor_id`, `cell_type`, `disease`, etc.
   - Applies normalization
        - **ATTENTION:** check the code for the differt types of normalization that can be perfomed.
   - Merges all datasets into a single `AnnData` object, ensuring missing genes are handled.
   - Creates concatenated labels (`cell_type_ontology_term_id-disease_ontology_term_id`) and encodes them as integers.
   - Removes cells with duplicated barcodes.
   - Saves the processed dataset as `merged_30.h5ad`.

## Training

The model to save time is trained using parallel programming. 
The model can be trained using CPUs or GPUs, in the latter case it can be parallelized using [DP](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) or [DDP](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html). A little summary:

- CPU: not use. Too Slow.
- GPU DDP: not use. Code is there, but not work (complicated to parallelize mutliple processes using GPUs).
- **GPU DP**: Please use this:
    - [model/VAE_parallel_gpu_dp.py](model/VAE_parallel_gpu_dp.py): contains the model architecture for this specific parallelization.
    - [model/train_parallel_gpu_dp.py](model/train_parallel_gpu_dp.py): code for the trainign of the model.
    - [scripts/3_VAE_train_gpu.run](scripts/3_VAE_train_gpu.run): bash script for running the code using SCITAS.

**Attention:** Before lunching please check in the script the hyperparamters (like the number of epochs).
**Attention:** The dataset is split in train and test always at the same way. Then the train set is further split in train and validation depeending on the variable `SEED_CROSSVALIDATION`. In this way, by runnign multiple times the script is it possbile to obtain a cross validation result. 































