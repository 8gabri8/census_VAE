import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import os

from utils import *


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=256):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 43201),  # input ~86402
            nn.ReLU(),  
            nn.Linear(43201, 21600),
            nn.ReLU(),
            nn.Linear(21600, 10800),
            nn.ReLU(),
            nn.Linear(10800, 4096),
            nn.ReLU(),
            nn.Linear(4096, latent_dim),  # Latent space
            nn.ReLU()
        )

        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10800),
            nn.ReLU(),
            nn.Linear(10800, 21600),
            nn.ReLU(),
            nn.Linear(21600, 43201),
            nn.ReLU(),
            nn.Linear(43201, input_size),  # Reconstruct the original input
            # Sigmoid or no activation depending on your data
        )

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        return encoded, decoded

def weight_init(m):
    if isinstance(m, nn.Linear):
        # He initialization for ReLU activations
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He uniform for ReLU
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Bias initialized to zero

class VAE(AutoEncoder):

    def __init__(self, input_size, latent_dim=256):
        super().__init__(input_size, latent_dim)
        
        # mu and log_var are simpleNN that will meant the mean and std from the latent space emb
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

        # In case is the initilaization the problem of wwigth explosion
        # self.mu.apply(weight_init)  # Apply initialization to mu layer
        # self.log_var.apply(weight_init)  # Apply initialization to log_var layer

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # random noise
        return mu + eps * std

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        
        # Compute the mean and log variance
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        
        # Reparameterize the latent vector
        z = self.reparameterize(mu, log_var)
        
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        
        # Return the encoded, decoded, and the mean/log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # Sample from the standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device) # put z on device of the mdoel
            # Pass the latent samples through the decoder
            samples = self.decoder(z)
        return samples
    
class VAEWithClassifier(VAE):
    def __init__(self, input_size, latent_dim=256, num_classes=10):
        super().__init__(input_size, latent_dim)
        
        # Define the classification NN
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer for classification
        )

        # Could create also two different classifiers, one for cell_type an one for disease
        # TODO:
        
    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        
        # Compute the mean and log variance for the VAE
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)

        # Attention:clipping
        log_var = torch.clamp(log_var, min=-5, max=5)
        
        # Reparameterize the latent vector
        z = self.reparameterize(mu, log_var)
        
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        
        # Classifier: Classify the latent space representation
        logits_classification = self.classifier(encoded)
        
        return encoded, decoded, mu, log_var, logits_classification
    
    def predict_from_logit(self, logits_classification):

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits_classification, dim=1)
        
        # Get the predicted class label (index of the max probability)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        return predicted_classes
    
    def predict_from_input(self, input):

        _, _, _, _, logits_classification = self.forward(input)
        
        return self.predict_from_logit(logits_classification)

def losses_VAE(
    recon_x, 
    x, 
    mu, 
    logvar, 
    classification_output, 
    labels,
    reconstruction_loss = "mse",
    weight_reconstruction=1.0, 
    weight_KLD=1.0, 
    weight_classification_loss=1.0,
    verbose = False,
):
    """
    Compute the combined losses for the VAE with classification: 
        - Reconstruction loss --> MSE
        - Normality fo latent space --> KL divergence 
        - Classification loss --> Cross Entropy
    
    Arguments:
    recon_x (tensor): The reconstructed input from the decoder.
    x (tensor): The original input data.
    mu (tensor): The mean of the latent variable distribution from the encoder.
    logvar (tensor): The log variance of the latent variable distribution from the encoder.
    classification_output (tensor): The classifier output.
    labels (tensor): The true labels for classification.
    
    Returns:
    total_loss (tensor): The combined, weighted loss value (normalized to 1).
    """

    # Reconstruction Loss
    if reconstruction_loss == "mse":
        mse_loss_fn = nn.MSELoss(reduction='mean')
        rec_loss = mse_loss_fn(recon_x, x) # Calculte the mean of each single sample loss
    elif reconstruction_loss == "bce":
        rec_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    
    # Normality for latent space --> KL divergence 
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


    # Classification loss --> Cross Entropy
    classification_loss = F.cross_entropy(classification_output, labels, reduction='mean')
    
    # Weighted losses
    rec_loss_weighted = weight_reconstruction * rec_loss
    KLD_weighted = weight_KLD * KLD
    classification_loss_weighted = weight_classification_loss * classification_loss
    
    # Total loss (sum of weighted losses)
    total_loss = rec_loss_weighted + KLD_weighted + classification_loss_weighted
        
    if verbose:
        print(f"Total Loss: {total_loss.item():.4f}\n"
            f"Reconstruction Loss (weighted): {rec_loss_weighted.item():.4f}\n"
            f"KLD (weighted): {KLD_weighted.item():.4f}\n"
            f"Classification Loss (weighted): {classification_loss_weighted.item():.4f}\n")
    
    return total_loss, rec_loss_weighted, KLD_weighted, classification_loss_weighted

def predict_labels(    
    model, 
    adata,
    batch_size=64,
    weigth_losses = [1, 0.01, 1]
):
    
    w_rec, w_kl, w_cls = weigth_losses


    # Automatically detect the device of the model
    device = next(model.parameters()).device

    # Save true adn predicted labels
    true_labels = []
    predicted_labels = []
    total_loss = 0

    with torch.no_grad():

        model.eval()

        # Create list with batch indices
        batch_indices = [list(range(i, min(i + batch_size, adata.shape[0]))) for i in range(0, adata.shape[0], batch_size)]

        # Iterate over the batches in the training data
        for single_batch_indices in tqdm(batch_indices, unit="batch"):
            
            # Ectract data and labels of this batch
            adata_tmp = adata[single_batch_indices, ]
            y = adata_tmp.obs["concat_label_encoded"].values.tolist()
            x = adata_tmp.X.toarray() #num_cells(batch_size) x num_genes

            # Move data and labels to the appropriate device
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            # Forward pass
            encoded, decoded, mu, log_var, logits_classification = model(x)

            # Calculate loss
            loss, BCE_loss_weighted, KLD_loss_weighted, classification_loss_weighted = losses_VAE(
                decoded, x, mu, log_var, logits_classification, y,
                weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
            )  

            # Update loss
            total_loss += loss

            # Apply Softmax to logits to get probabilities
            probabilities = torch.softmax(logits_classification, dim=1)

            # Predict the class with the highest probability
            batch_predicted_labels = torch.argmax(probabilities, dim=1)

            # Store predictions
            predicted_labels.extend(batch_predicted_labels.cpu().numpy().tolist())  # Convert to list of ints
            true_labels.extend(y.cpu().numpy().tolist())  # Convert to list of ints

    # Normalize loss
    total_loss /= adata.obs.shape[0]

    return true_labels, predicted_labels, total_loss

def train_in_parallel(
    rank, 
    world_size, 
    n_genes, 
    n_classes, 
    adata_train, 
    batch_size, 
    num_epochs, 
    model_out_queue,
    weigth_losses = [1, 0.01, 1],
    lr=1e-4,
):
    
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Set this to the IP address of the master node
    os.environ["MASTER_PORT"] = "29500"      # Set this to an open port on the master node
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize the distributed process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    print(f"enter in train {rank}")

    w_rec, w_kl, w_cls = weigth_losses

    print(f"{rank} - start process:", log_memory_usage())

    # Instantiate the model --> reinitialize each time
    model = VAEWithClassifier(
        input_size = n_genes,
        latent_dim=256, 
        num_classes=n_classes
    )

    print(f"{rank} - create model:", log_memory_usage())


    # device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)
    # model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model)#, device_ids=[rank])

    # Create a DataLoader with DistributedSampler
    dataset = AnnDataDataset(adata_train, device="cpu")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Save losses of train and val for process rank=0
    epoch_losses_train_rank_0 = []
    epoch_losses_val_rank_0 = []


    print(f"{rank} - before training loop:", log_memory_usage())

    # Training loop
    model.train()
    for epoch in range(num_epochs):

        
        sampler.set_epoch(epoch)  # Make sure the sampler knows the epoch
        running_loss = 0.0

        # Initialize tqdm only for rank 0
        if rank == 0:
            # Create the progress bar for the first rank
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        else:
            # Otherwise, set progress_bar to a dummy iterator for other ranks
            progress_bar = dataloader

        print(f"epoch {epoch} {rank}")


        for inputs, targets in progress_bar:

            inputs = inputs.squeeze(1)

            optimizer.zero_grad()

            encoded, decoded, mu, log_var, logits_classification = model(inputs)

            print(f"{rank} - After forward:", log_memory_usage())

            #print(logits_classification, logits_classification.shape)

            loss, BCE_loss_weighted, KLD_loss_weighted, classification_loss_weighted = losses_VAE(
                decoded, inputs, mu, log_var, logits_classification, targets,
                weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
            )  

            print(f"{rank} -After loss:", log_memory_usage())

            print(f"Performing Backward {rank}")
            loss.backward() #
            optimizer.step()
            print(f"Weigths updated {rank}")

            running_loss += loss.item()

        # Print and Save Train loss of this epoch --> only for process rank = 0
        if rank == 0:
            epoch_loss_train = running_loss / len(dataloader) # Total loss / number of examples
            epoch_losses_train_rank_0.append(epoch_loss_train)   
            print(f"Epoch {epoch}, Loss: {running_loss / len(dataloader)}")

        # Save loss on Val set of this epoch --> only for process rank = 0
        if rank == 0:
            pass #evaluate on validation

    if rank == 0:
        model_out_queue.put(model)  # Put the model in the queue from rank 0

    # Clean up
    dist.destroy_process_group()

