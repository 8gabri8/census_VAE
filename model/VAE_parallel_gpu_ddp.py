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
from torch.nn.parallel import DistributedDataParallel as DDP


from utils import *


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=256):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 21600),  # input ~86402
            # nn.ReLU(),  
            # nn.Linear(43201, 21600),
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
            # nn.Linear(21600, 43201),
            # nn.ReLU(),
            nn.Linear(21600, input_size),  # Reconstruct the original input
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

        # Message to see if gpu parallelism is working
        #print("\tIn Model (single gpu): input size", x.size(), "output size", logits_classification.size())

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

def calculate_loss_labels(    
    model, 
    adata,
    batch_size=64,
    weigth_losses = [1, 0.01, 1]
):
    
    # Automatically detect the device of the model
    device = next(model.parameters()).device
    
    # Unpack losses weights
    w_rec, w_kl, w_cls = weigth_losses

    # Create Dataloader
    adata_dataset = AnnDataDataset(adata)
    dataloader = DataLoader(adata_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    true_labels = []
    predicted_labels = []

    # Evaluation loop
    model.eval()
    for inputs, targets in tqdm(dataloader, desc=f"Batch", dynamic_ncols=True):

        # Put values on device
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.squeeze(1)
        targets = targets.squeeze(1)

        # Forward pass
        encoded, decoded, mu, log_var, logits_classification = model(inputs) ## The model will automatically use multiple GPUs if DataParallel is used

        # Calculate loss
        loss, BCE_loss_weighted, KLD_loss_weighted, classification_loss_weighted = losses_VAE(
            decoded, inputs, mu, log_var, logits_classification, targets,
            weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
        )  

        # Update loss
        total_loss += loss.item()

        # Apply Softmax to logits to get probabilities
        probabilities = torch.softmax(logits_classification, dim=1)

        # Predict the class with the highest probability
        batch_predicted_labels = torch.argmax(probabilities, dim=1)

        # Store predictions
        predicted_labels.extend(batch_predicted_labels.cpu().numpy().tolist())  # Convert to list of ints
        true_labels.extend(targets.cpu().numpy().tolist())  # Convert to list of ints

        del inputs, targets, encoded, decoded, mu, log_var, logits_classification, batch_predicted_labels
        torch.cuda.empty_cache()

    # Normalize loss
    total_loss /= len(dataloader)

    return true_labels, predicted_labels, total_loss

def train_parallel_gpu_ddp(  
    rank,
    world_size,
    n_genes, 
    n_classes, 
    adata_train, 
    adata_val,
    batch_size, 
    num_epochs, 
    model_out_queue,
    weigth_losses = [1, 0.01, 1],
    lr=1e-4,
):
    
    print(f"Running DDP train on rank {rank}.")

    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Set this to the IP address of the master node
    os.environ["MASTER_PORT"] = "29500"      # Set this to an open port on the master node
    #os.environ["WORLD_SIZE"] = str(world_size)
    #os.environ["RANK"] = str(rank)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # Initialize the distributed process group
    # nccl --> GPU
    # gloo --> CPU
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Unpack losses weights
    w_rec, w_kl, w_cls = weigth_losses

    # Instantiate the model --> reinitialize each time
    model = VAEWithClassifier(
        input_size = n_genes,
        latent_dim=256, 
        num_classes=n_classes
    )

    model.to(rank)

    # Create DDP istance
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    # Create Dataloader
    adata_train_dataset = AnnDataDataset(adata_train)
    sampler = torch.utils.data.distributed.DistributedSampler(adata_train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(adata_train_dataset, batch_size=batch_size, sampler=sampler)

    # Initialize optimizer
    #optimizer = optim.Adam(ddp_model.parameters(), lr=lr)
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)


    # Save losses foe eahc epoch (save only for rank 0)
    losses_train_rank_0 = []
    losses_val_rank_0 = []

    # Training loop
    for epoch in range(num_epochs):

        train_dataloader.sampler.set_epoch(epoch)  # Make sure the sampler knows the epoch
        this_epoch_loss = 0.0

        # Initialize tqdm only for rank 0
        if rank == 0:
            # Create the progress bar for the first rank
            progress_bar = tqdm(train_dataloader, desc=f"rank {rank} - Epoch {epoch}", dynamic_ncols=True)
        else:
            # Otherwise, set progress_bar to a dummy iterator for other ranks
            progress_bar = train_dataloader

        model.train()
        for inputs, targets in progress_bar:

            print("hello")
            # Synchronize all processes to make sure they are ready to start training
            dist.barrier()  # All processes wait here

            # Put values on device
            inputs, targets = inputs.to(rank), targets.to(rank)

            inputs = inputs.squeeze(1)
            targets = targets.squeeze(1)
            #print("\n", inputs.shape, targets.shape)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            encoded, decoded, mu, log_var, logits_classification = ddp_model(inputs) ## The model will automatically use multiple GPUs if DataParallel is used

            # Check that computation has been split in differ gpu in parallelization
            #print("Outside: input size", inputs.size(), "output_size", logits_classification.size())

            # Calculate loss
            loss, BCE_loss_weighted, KLD_loss_weighted, classification_loss_weighted = losses_VAE(
                decoded, inputs, mu, log_var, logits_classification, targets,
                weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
            )  

            #print("Before backward", torch.cuda.memory_summary())

            # Backward pass (calculte gradients)
            loss.backward() 

            #print("After backward", torch.cuda.memory_summary()) # check the peak usage

            # Update weigths
            optimizer.step()

            #print("After step", torch.cuda.memory_summary()) # check the peak usage

            # Add loss of this batch to the loss of this epoch
            this_epoch_loss += loss.item()

            del inputs, targets, encoded, decoded, mu, log_var, logits_classification
            torch.cuda.empty_cache()

            #print("end batch", torch.cuda.memory_summary()) # check the peak usage

        print(f"end epoch - {rank}")
        # Synchronize all processes to make sure they are ready to start training
        dist.barrier()  # All processes wait here


        # Only for process rank=0
        #   1) Save loss of this epoch
        #   2) Evaluate on Val set at the end of the epoch
        # if rank == 0:
        #     epoch_loss_train = this_epoch_loss / len(train_dataloader) # Total loss / number of examples
        #     losses_train_rank_0.append(epoch_loss_train)   
        #     print(f"Epoch {epoch}, Loss: {epoch_loss_train}")

        #     # compute for VALIDATION set
        #     print("\nCompute Loss for Validation Set:")
        #     _, _, this_epoch_loss_validation = calculate_loss_labels(    
        #                                         ddp_model, 
        #                                         adata_val,
        #                                         batch_size=batch_size,
        #                                         weigth_losses = weigth_losses
        #     )
        #     losses_val_rank_0.append(this_epoch_loss_validation) 

    # Save final model (put in the queue)
    if rank == 0:
        model_out_queue.put(model)  
        model_out_queue.put(losses_train_rank_0)  
        model_out_queue.put(losses_val_rank_0)  

    # Clean up
    dist.destroy_process_group()









