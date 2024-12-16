import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from VAE import *

# Remember to checj in the outer.run file that the memory and time are appropriate for the task

#########################
### Setting Device
#########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#if device == "cpu":
n_cpus = os.cpu_count()
print("Number of cores: ", n_cpus )

# Set the number of threads for MKL and OpenMP
# ATTENTION: too many can take even more time
# os.environ['OMP_NUM_THREADS'] = str(n_cpus) # Set to number of CPU cores you want to use
# os.environ['MKL_NUM_THREADS'] = str(n_cpus) # Same as OMP for MKL operations
# torch.set_num_threads(n_cpus) # Set the number of threads for intra-op parallelism (operations within a layer)
# torch.set_num_interop_threads(n_cpus) # Set the number of threads for inter-op parallelism (operations between layers)


#########################
### Set Random Seed
#########################

np.random.seed(42)
torch.manual_seed(42)

#########################
### Read from argparse
#########################

parser = argparse.ArgumentParser(description="Train VAE model")
parser.add_argument('--job_id', type=str, required=True, help="job_id")
args = parser.parse_args()

job_id = args.job_id

#########################
### Load h5ad
#########################

adata = sc.read("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_log1.h5ad")
# adata = adata[:1000,]
# adata.write("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_log1_1000_cells.h5ad")
#adata = sc.read("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_log1_1000_cells.h5ad")

# Number of cells
n_cells = adata.obs.shape[0]
# Number of Genes
n_genes = adata.var.shape[0]
# Number fo classes to predict
n_classes = len(adata.obs["concat_label_encoded"].unique())
#n_classes = 265 # ATTENTION: only when using smaller dataset, otherwise CrossEntirpy will give error
# Number of Epcohs
num_epochs = 1
# Batch size
batch_size = 512
# Number of folds for crossvalidation
n_folds = 1
# wirghts of the differt losses of VAE
weigth_losses = [1, 0.01, 1] #reconstruction, kl, classification
# Folder where to save logs and results
SAVE_FOLDER = f"/scratch/dalai/results_census/3_VAE_train_{job_id}"
os.makedirs(SAVE_FOLDER, exist_ok=True)

print(f"""
n_cells: {n_cells}
n_genes: {n_genes}
n_classes: {n_classes}
batch size: {batch_size}
n epochs: {num_epochs}
n folds: {n_folds}
save folder: {SAVE_FOLDER}\n
""")

#########################
### Split in train and test
#########################

indices = np.random.permutation(n_cells)

# Split indices for 10% and 90%
split_10 = int(n_cells * 0.1) #numer foc ells correspfing to 10%
indices_10 = indices[:split_10]
indices_90 = indices[split_10:]

# Subset AnnData object
adata_test = adata[indices_10].copy()
adata_train = adata[indices_90].copy()

print("Test AnnData shape:", adata_test.shape)
print("Train AnnData shape:", adata_train.shape)

n_cells_train = adata_train.shape[0]

print(f"""
For Train Set:
n_cells: {n_cells_train}
n_genes: {n_genes}
n_classes: {n_classes}
batch size: {batch_size}
n epochs: {num_epochs}
n folds: {n_folds}
save folder: {SAVE_FOLDER}\n
""")

#########################
### Implement Cross Validation
#########################

# Initialize a list to store the indices for different splits
splits_indices = []
    # Attention: it is a list of tuple. each tuple has 2 elemets: one is a list of the indiced of sample in the train, and the other is list woith the indices of the samples on the test

# Set the number of splits you want, here we'll use a single train/test split
for _ in range(n_folds):
    # Perform train/val split
    train_indices, val_indices = train_test_split(range(n_cells_train), test_size=0.2, random_state=None)
    
    # Store the indices for each split in the list
    splits_indices.append((train_indices, val_indices))

print(f"Total number of Folds: {len(splits_indices)}")
train_indices, val_indices = splits_indices[0]
print(f"Training indices shape: {len(train_indices)}")
print(f"Validation indices shape: {len(val_indices)}")
print(f"Test indices shape: {adata_test.shape[0]}")

#########################
### Train
#########################

# Save results for each Fold
models = [] # each elemtn is a trained model from fold
losses_train = []
losses_val = [] # each element is a list with the losses of a fold

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

    # Attention
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
    #torch.autograd.set_detect_anomaly(True)

    # create adata_val
    train_indices, val_indices = fold
    adata_val = adata_train[val_indices,].copy()

    # Call Training loop
    model, epoch_losses_train, epoch_losses_val = train_VAE_with_classification(
        model = model, 
        fold_indices = fold, 
        lr=1e-4, 
        num_epochs=num_epochs,
        adata = adata_train, # will be split inside in train and val
        batch_size=batch_size,
        weigth_losses = weigth_losses #reconstruction, kl, classification
    )

    # Save losses
    models.append(model)
    losses_train.append(epoch_losses_train)
    losses_val.append(epoch_losses_val)

    # Save image losses
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses_train, label='Training Loss', color='blue', marker='o')
    plt.plot(epoch_losses_val, label='Validation Loss', color='red', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    image_path = os.path.join(SAVE_FOLDER, f'{i}_losses_plot.png')
    plt.savefig(image_path)
    print(f"Loasses of fold {i} saved.")

    # Save model
    # model_path = os.path.join(SAVE_FOLDER, f'{i}_trained_model.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f"Model of fold {i} saved.")

    # Save the labels
    print("Predicting labels with trained model in TRAIN set.")
    true_train_labels, predicted_train_labels = predict_labels(model, adata_train, batch_size)
    train_accuracy = accuracy_score(true_train_labels, predicted_train_labels)
    print("Predicting labels with trained model in VAL set.")
    true_val_labels, predicted_val_labels = predict_labels(model, adata_val, batch_size)
    val_accuracy = accuracy_score(true_val_labels, predicted_val_labels)
    print("Predicting labels with trained model in TEST set.")
    true_test_labels, predicted_test_labels = predict_labels(model, adata_test, batch_size)
    test_accuracy = accuracy_score(true_test_labels, predicted_test_labels)
    print(f"Accuracy on TRAIN set: {train_accuracy * 100:.2f}%")
    print(f"Accuracy on VAL set: {val_accuracy * 100:.2f}%")
    print(f"Accuracy on TEST set: {test_accuracy * 100:.2f}%")

    labels_data = {
        "train_accuracy": train_accuracy, 
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "true_train_labels": true_train_labels,
        "predicted_train_labels": predicted_train_labels,
        "true_val_labels": true_val_labels,
        "predicted_val_labels": predicted_val_labels,
        "true_test_labels": true_test_labels,
        "predicted_test_labels": predicted_test_labels}
    predicted_labels_path =  os.path.join(SAVE_FOLDER, f'{i}_labels_predicted.json')
    with open(predicted_labels_path, "w") as f:
        json.dump(labels_data, f, indent=4)
    print(f"\nLabels saved to {predicted_labels_path}")

    # Save confusion matrices
    # cm_train = confusion_matrix(true_train_labels, predicted_train_labels, labels=range(n_classes)) #Takes a lot of time with so many classes
    # cm_val = confusion_matrix(true_val_labels, predicted_val_labels, labels=range(n_classes))
    # cm_test = confusion_matrix(true_test_labels, predicted_test_labels, labels=range(n_classes))
    # fig, axes = plt.subplots(3, 1, figsize=(100, 300))
    # plot_confusion_matrix(cm_train, axes[0], "Training Set")
    # plot_confusion_matrix(cm_val, axes[1], "Validation Set")
    # plot_confusion_matrix(cm_test, axes[2], "Test Set")
    # plt.tight_layout()
    # confusion_matrix_path = os.path.join(SAVE_FOLDER, f"{i}_confusion_matrices.png")
    # plt.savefig(confusion_matrix_path)
    # plt.close()
    # print(f"\nConfusion matrices saved to {confusion_matrix_path}")



    # Load the state dictionary
    #model.load_state_dict(torch.load(save_path))
