import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt


from VAE import *

#########################
### Setting Device
#########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#########################
### Set Random Seed
#########################

np.random.seed(42)
torch.manual_seed(42)

#########################
### Load h5ad
#########################

adata = sc.read("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30.h5ad")
print(type(adata.X))
adata = adata[:100,]
adata.write("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_100_cells.h5ad")
#adata = sc.read("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_100_cells.h5ad")
# Number of cells
n_cells = adata.obs.shape[0]
# Number of Genes
n_genes = adata.var.shape[0]
# Number fo classes to predict
n_classes = len(adata.obs["concat_label_encoded"].unique())
n_classes = 265 # ATTENTION: only when using smaller dataset, otherwise CrossEntirpy will give error
# Number of Epcohs
num_epochs = 5

print(f"n_cells: {n_cells}, n_genes: {n_genes}, n_classes: {n_classes}")

# Extraction of values(single cell expression + labels) will be done isde the train loop, in order to put in a dense format only the current batch

#########################
### Implement Cross Validation
#########################

# Number of folds 
n_folds = 5

# Initialize a list to store the indices for different splits
splits_indices = []

# Set the number of splits you want, here we'll use a single train/test split
for _ in range(n_folds):
    # Perform train/test split
    train_indices, val_indices = train_test_split(range(n_cells), test_size=0.2, random_state=None)
    
    # Store the indices for each split in the list
    splits_indices.append((train_indices, val_indices))

print(f"Total number of Folds: {len(splits_indices)}")
train_indices, val_indices = splits_indices[0]
print(f"Training indices shape: {len(train_indices)}")
print(f"Validation indices shape: {len(val_indices)}")

#########################
### Instantiate the model
#########################

model = VAEWithClassifier(
    input_size = n_genes,
    latent_dim=256, 
    num_classes=n_classes
)

model.to(device)

# Attention
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#########################
### Train
#########################

models = [] # each elemtn is a trained model from fold
losses_train = []
losses_test = [] # each element is a list with the losses of a fold

# Perform Cross Validation
for i, fold in enumerate(splits_indices):

    print(f"\nCross Validation, fold {i+1}/{n_folds}")

    # Instantiate the model --> reinitialize each time
    model = VAEWithClassifier(
        input_size = n_genes,
        latent_dim=256, 
        num_classes=n_classes
    )
    model.to(device)

    # Call Training loop
    model, epoch_losses_train, epoch_losses_test = train_VAE_with_classification(
        model = model, 
        fold_indices = fold, 
        lr=1e-3, 
        num_epochs=num_epochs,
        adata = adata,
        batch_size=10
    )

    models.append(model)
    losses_train.append(epoch_losses_train)
    losses_test.append(epoch_losses_test)

    # Save image
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses_train, label='Training Loss', color='blue', marker='o')
    plt.plot(epoch_losses_test, label='Testing Loss', color='red', marker='o')
    plt.title('Training and Testing Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    image_path = f'/scratch/dalai/census_VAE/log/{i}_losses_plot.png'
    plt.savefig(image_path)

