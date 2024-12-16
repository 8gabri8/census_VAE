import seaborn as sns
import copy
from torch.utils.data import Dataset
import torch


def plot_confusion_matrix(cm, ax, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

class EarlyStopping:
    def __init__(self, max_patience=5):
        self.best_loss = float('inf')  # Initialize the best loss to a very high value
        self.best_model_weights = None  # Store the model weights of the best model
        self.max_patience = max_patience  # Maximum number of patience epochs
        self.patience = max_patience  # Initialize patience counter
        self.early_stop = False  # Early stopping flag

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:  # If current loss is better than the best
            self.best_loss = val_loss  # Update the best loss
            self.best_model_weights = copy.deepcopy(model.state_dict())  # Save the best model weights
            self.patience = self.max_patience  # Reset patience counter
        else:
            self.patience -= 1  # Decrease patience counter
            if self.patience == 0:  # If patience runs out, trigger early stop
                self.early_stop = True

# Define a simple dataset
class AnnDataDataset(Dataset):
    def __init__(self, adata, device):
        self.device = device
        self.adata = adata

    def __len__(self):
        return self.adata.obs.shape[0]

    def __getitem__(self, idx):
        adata_tmp = self.adata[idx, ]
        y = adata_tmp.obs["concat_label_encoded"].values.tolist()
        x = adata_tmp.X.toarray() #num_cells(batch_size) x num_genes

        # Move data and labels to the appropriate device
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        y = y.squeeze(-1)  # Removes the last dimension if it is 1

        return x, y

import psutil
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    #print(f"Memory Usage: {mem_info.rss / (1024 ** 2)} MB")
    return mem_info.rss / (1024 ** 3) # in GB