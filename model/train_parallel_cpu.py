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

from model.VAE_parallel_cpu import *

# Remember to checj in the outer.run file that the memory and time are appropriate for the task

def main():
    #########################
    ### Setting Device
    #########################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_cpus = os.cpu_count()
    print("Number of cores: ", n_cpus )
    WORLD_SIZE = 3# n_cpus/processes in parallel

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

    #adata = sc.read("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_log1.h5ad")
    # adata = adata[:1000,]
    # adata.write("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_log1_1000_cells.h5ad")
    adata = sc.read("/work/upcourtine/clock-classifier/gabriele-results/census_results/merged_30_log1_1000_cells.h5ad")

    # Number of cells
    n_cells = adata.obs.shape[0]
    # Number of Genes
    n_genes = adata.var.shape[0]
    # Number fo classes to predict
    n_classes = len(adata.obs["concat_label_encoded"].unique())
    n_classes = 265 # ATTENTION: only when using smaller dataset, otherwise CrossEntirpy will give error
    # Number of Epcohs
    num_epochs = 1
    # Batch size
    batch_size = 10 #512 #10
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

        model_out_queue = mp.Queue()

        # create adata_val (specific for this fold)
        train_indices, val_indices = fold
        adata_train_fold = adata_train[train_indices,].copy()
        adata_val = adata_train[val_indices,].copy()

        # Call Training loop
        print("Start Train in Parallel...")
        mp.spawn(
            train_in_parallel, 
            args=(
                # RANK AOUTOMAICALLY PASSED
                WORLD_SIZE,  # Pass world_size as the first argument
                n_genes, 
                n_classes, 
                adata_train_fold, 
                batch_size, 
                num_epochs, 
                model_out_queue,
                weigth_losses,  # Pass weight_losses as a list
                1e-4  # Pass learning rate
            ), 
            nprocs=WORLD_SIZE, 
            join=True
        )

        model = model_out_queue.get()  # Model is returned from rank 0 after all processes finish

        # Save losses
        models.append(model)

        # Save model
        # model_path = os.path.join(SAVE_FOLDER, f'{i}_trained_model.pth')
        # torch.save(model.state_dict(), model_path)
        # print(f"Model of fold {i} saved.")

        # Save the labels
        print("Predicting labels with trained model in TRAIN set.")
        true_train_labels, predicted_train_labels, loss_train = predict_labels(model, adata_train, batch_size, weigth_losses)
        train_accuracy = accuracy_score(true_train_labels, predicted_train_labels)
        print("Predicting labels with trained model in VAL set.")
        true_val_labels, predicted_val_labels, loss_val = predict_labels(model, adata_val, batch_size, weigth_losses)
        val_accuracy = accuracy_score(true_val_labels, predicted_val_labels)
        print("Predicting labels with trained model in TEST set.")
        true_test_labels, predicted_test_labels, loss_test = predict_labels(model, adata_test, batch_size, weigth_losses)
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


if __name__ == '__main__':
    main()
    print("The trained model is ready for use.")