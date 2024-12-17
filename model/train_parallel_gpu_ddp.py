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

from VAE_parallel_gpu_ddp import *

# Remember to checj in the outer.run file that the memory and time are appropriate for the task

def main():
    #########################
    ### Setting Device
    #########################

    # See device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check how many GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    
    # Check how mnay CPUs
    n_cpus = os.cpu_count()
    print("Number of cores/CPUs: ", n_cpus )

    # Check total RAM
    total_memory = psutil.virtual_memory().total
    total_memory_gb = total_memory / (1024 ** 3)
    print(f"Total System RAM: {total_memory_gb:.2f} GB")

    # Check GPUs free sapce
    #print(torch.cuda.memory_summary())
    num_gpus=2

    #########################
    ### Set Random Seed
    #########################

    np.random.seed(42)
    torch.manual_seed(42)
    SEED_CROSSVALIDATION = 0

    #########################
    ### Read from argparse
    #########################

    parser = argparse.ArgumentParser(description="Train VAE model")

    parser.add_argument('--job_id', type=str, required=True, help="job_id")
    parser.add_argument('--seed_crossvalidation', type=str, required=True, help="seed_crossvalidation")

    args = parser.parse_args()

    job_id = args.job_id
    SEED_CROSSVALIDATION = args.seed_crossvalidation

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
    batch_size = 100 #512 #100
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
    ### Split in train, Val, test
    #########################

    # Random permutation of the indices of all cells
    indices = np.random.permutation(n_cells)

    # Split indices for 10% and 90% --> ATTENTION: splitting in train and test always the same, independt of SEED_CROSSVALIDATION
    split_10 = int(n_cells * 0.1) #numer foc ells correspfing to 10%
    indices_10 = indices[:split_10]
    indices_90 = indices[split_10:]

    # Subset AnnData object
    adata_test = adata[indices_10].copy()
    adata_train = adata[indices_90].copy()

    # Split for validation --> ATTENTION: splitting in train val depends on SEED_CROSSVALIDATION
    n_cells_train = adata_train.obs.shape[0]
    np.random.seed(int(SEED_CROSSVALIDATION)) # chnage seed
    indices_train = np.random.permutation(n_cells_train)
    split_10 = int(n_cells_train * 0.1)
    indices_10 = indices_train[:split_10]
    indices_90 = indices_train[split_10:]
    adata_val = adata_train[indices_10].copy()
    adata_train = adata_train[indices_90].copy()

    print("Train AnnData shape:", adata_train.shape)
    print("Validation AnnData shape:", adata_val.shape)
    print("Test AnnData shape:", adata_test.shape)

    del adata # no more useful

    #########################
    ### Train
    #########################

    # List to retrieve data from the parallel processes
    model_out_queue = mp.Queue()

    # Call Training loop
    print("\nStart Train in Parallel...\nLaunching parallel processes")
    mp.spawn(
        train_parallel_gpu_ddp, 
        args=(
            # RANK AOUTOMAICALLY PASSED
            num_gpus,  # Pass world_size as the first argument
            n_genes, n_classes, # Needed to intialize the model
            adata_train, 
            adata_val, 
            batch_size, 
            num_epochs, 
            model_out_queue,
            weigth_losses,  
            1e-4  # Pass learning rate
        ), 
        nprocs=num_gpus, 
        join=True
    )

    model = model_out_queue.get()  # Model is returned from rank 0 after all processes finish
    losses_train = model_out_queue.get()
    losses_val = model_out_queue.get()


    #########################
    ### Predict Labels
    #########################

    print("Predicting labels with trained model in TRAIN set.")
    true_train_labels, predicted_train_labels, _ = calculate_loss_labels(model, adata_train, batch_size, weigth_losses)
    train_accuracy = accuracy_score(true_train_labels, predicted_train_labels)

    print("Predicting labels with trained model in VAL set.")
    true_val_labels, predicted_val_labels, _  = calculate_loss_labels(model, adata_val, batch_size, weigth_losses)
    val_accuracy = accuracy_score(true_val_labels, predicted_val_labels)

    print("Predicting labels with trained model in TEST set.")
    true_test_labels, predicted_test_labels, _  = calculate_loss_labels(model, adata_test, batch_size, weigth_losses)
    test_accuracy = accuracy_score(true_test_labels, predicted_test_labels)

    print(f"Accuracy on TRAIN set: {train_accuracy * 100:.2f}%")
    print(f"Accuracy on VAL set: {val_accuracy * 100:.2f}%")
    print(f"Accuracy on TEST set: {test_accuracy * 100:.2f}%")

    #########################
    ### Save Results
    #########################

    # Save image losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses_train, label='Training Loss', color='blue', marker='o')
    plt.plot(losses_val, label='Validation Loss', color='red', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    image_path = os.path.join(SAVE_FOLDER, f'{i}_losses_plot.png')
    plt.savefig(image_path)
    print(f"Losses of fold {i} saved.")

    # Save model
    # model_path = os.path.join(SAVE_FOLDER, f'{i}_trained_model.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f"Model of fold {i} saved.")

    # Load the state dictionary
    #model.load_state_dict(torch.load(save_path))

    # Save the labels
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


if __name__ == '__main__':
    main()