#########################
### Libraries
#########################

import os
import pandas as pd
import scanpy as sc
import os
import numpy as np
import anndata as ad
from anndata import AnnData
from anndata import read_h5ad
import gc  

print("Libraries uploaded.\n")

DATASET_DIR = "/work/upcourtine/clock-classifier/gabriele-results/census_results/datasets"

#########################
### Find Dataset Path
#########################

# Find paths of all datasets
dataset_paths = []
for file_name in os.listdir(DATASET_DIR):
    if file_name.endswith('.h5ad'):
        file_path = os.path.join(DATASET_DIR, file_name)
        if os.path.isfile(file_path):  # Check if it's a file
            dataset_paths.append(file_path)

# Columns of .obs that each df must have
must_columns = ["donor_id", 
                "cell_type", "cell_type_ontology_term_id",
                "disease", "disease_ontology_term_id", 
                "tissue", "tissue_ontology_term_id",
                "assay", "assay_ontology_term_id"]

#########################
### Check all datasets have must columns
#########################

# Checj that each df has the "must_columns"
print(f"Check all dtasets have must columns.\n")
for i, dataset in enumerate(dataset_paths):
    #print(f"{i} ", dataset)

    # Read the .h5ad file using scanpy
    adata = sc.read(dataset)

    # Check that the necessary columns are present
    missing_columns = [col for col in must_columns if col not in adata.obs.columns]
    if missing_columns:
        print(f"Missing the following required columns: {', '.join(missing_columns)}\n")

    # just to make some checks 
    if (adata.obs["organism"] != "Homo sapiens").any():
        print("Non-humans cells detected")
    if (adata.obs["is_primary_data"] == False).any():
        print("Non Primary cells detected")

#########################
### Read dataframe with gene names mapping
#########################

df_genes_map = pd.read_csv( os.path.join("/home/dalai/git/clock-classifier/Gabriele/assets", "mapping_genes_census.csv"), index_col = False)

# `gene_to_keep` is the list of genes you want to keep --> genes in all datasets
gene_to_keep = df_genes_map["gene_id"]

#########################
### Create a list with all datasets, Remove: not preset genes + Useless information
#########################

adata_datasets_list = []

# Remove genes not in list
# + 
# Remove useless information
for i, dataset in enumerate(dataset_paths):
    
    print(f"\n{dataset}")
    print(f"Dataset {i+1}/{len(dataset_paths)}")
    
    # Read AnnData
    adata = sc.read(dataset)

    #display(adata.var)

    # Extract the genes in the AnnData object
    genes_in_adata = adata.var.index #Series
    genes_in_adata_to_keep = genes_in_adata.isin(gene_to_keep)
    #print(genes_in_adata)

    # Maintain only the genes that must be kept
    adata = adata[:, genes_in_adata_to_keep]

    # Shrink the dataset, i.e. mantain only
        # raw count matrix
        # .var
        # subset of .obs

    # Retain only the necessary columns in `obs`
    adata.obs = adata.obs[must_columns]

    # Remove all keys from `uns` (unstructured annotations)
    adata.uns = {}

    # Remove all keys from `obsm` (observations in multi-dimensional space)
    adata.obsm = {}

    adata_datasets_list.append(adata)

    # Clear memory for the next iteration
    del adata
    gc.collect()

# Check that we did not remoed anything important (X matrix is full)
print(adata_datasets_list[0])
print(adata_datasets_list[0].X)
# Check how many datset have been processed
len(adata_datasets_list)

#########################
### Normalize datasets (remove sequencing depth counfounder)
#########################

print()
for i, adata in enumerate(adata_datasets_list):
    print(f"Normalizing dataset {i}...")
    # Step 1: Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Step 2: Log-transform the data (optional but common)
    sc.pp.log1p(adata)

# little check
# Check the sum of counts per cell
total_counts_per_cell = np.sum(adata_datasets_list[0].X, axis=1)
print(f"Total counts per cell: {total_counts_per_cell[:10]}")

#########################
### Create a single dataset, Attention: be sure to have all ~82000 genens
#########################

# Merge all datasets
# +
# Add missing expression values for mssing genes for each dataset
    # easy: if we merge using "outer", by defiuakt missing genes will be put to 0

# Create an empty AnnData object with all genes from df_genes_map
all_genes_anndata = AnnData(
    X=np.zeros((0, df_genes_map.shape[0])),  # 0 "dummy" observation with zeros for all genes
    var=pd.DataFrame(index=df_genes_map["gene_id"])  # Set var to have all genes from df_genes_map
)

# Ensure all genes in df_genes_map are added
#print(f"All genes AnnData object: {all_genes_anndata}")

# Append the empty AnnData object to the list of datasets
adata_datasets_list.append(all_genes_anndata)

# Merge all datasets using `outer` join to include all genes
big_adata = ad.concat(adata_datasets_list, join="outer", label="orig_dataset", fill_value=0)

print(f"big_adata.obs: {big_adata.obs}")  # Displaying the observations (rows) of the data
print("big_adata.var:", big_adata.var)  # Displaying the variables (columns) of the data
print(f"big_adata.X.shape: {big_adata.X.shape}")  # Displaying the shape of the data matrix
print("big_adata.X: ", big_adata.X)  # Displaying the actual data matrix

#########################
### Add gene_name in .var
#########################

# Create a mapping from Ensembl IDs to Gene Names
name_mapping = dict(zip(df_genes_map['gene_id'], df_genes_map['gene_name']))

big_adata.var['gene_names'] = big_adata.var_names.map(name_mapping)

#########################
### Remove cells with duplicated barcode
#########################

duplicates = big_adata.obs_names[big_adata.obs_names.duplicated()]
#print(duplicates)

big_adata.obs_names_make_unique() #modifies in place

duplicates = big_adata.obs_names[big_adata.obs_names.duplicated()]
#print(duplicates) #it shodul be empty

#########################
### Create Labels
#########################

# Labels = concatenation of cell_type_id + disease_id
# NB already checked that id are the same across different datasets
# Also encoded labels are created

big_adata.obs["concat_label"] = big_adata.obs["cell_type_ontology_term_id"].astype(str) + '-' + big_adata.obs["disease_ontology_term_id"].astype(str)
print("In total the differtn labels are: ", big_adata.obs["concat_label"].unique().shape)

# Encode Labels as int
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
big_adata.obs["concat_label_encoded"] = label_encoder.fit_transform(big_adata.obs["concat_label"])

#big_adata.obs["concat_label_encoded"].sort_values()

#########################
### Save
#########################

big_adata.write("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30.h5ad")

print("Dataset Saved.")





