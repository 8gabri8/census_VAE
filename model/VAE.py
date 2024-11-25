import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F


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


class VAE(AutoEncoder):
    def __init__(self, input_size, latent_dim=256):
        super().__init__(input_size, latent_dim)
        
        # Define mu and log_var for reparameterization trick
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

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
            z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
            # Pass the latent samples through the decoder
            samples = self.decoder(z)
        return samples
    
class VAEWithClassifier(VAE):
    def __init__(self, input_size, latent_dim=256, num_classes=10):
        super().__init__(input_size, latent_dim)
        
        # Define the classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer for classification
        )
        
    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        
        # Compute the mean and log variance for the VAE
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        
        # Reparameterize the latent vector
        z = self.reparameterize(mu, log_var)
        
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        
        # Classifier: Classify the latent space representation
        classification_output = self.classifier(encoded)
        
        return encoded, decoded, mu, log_var, classification_output

def loss_BCE_KLD_classification(recon_x, x, mu, logvar, classification_output, labels,
                                 weight_BCE=1.0, weight_KLD=1.0, weight_classification_loss=1.0):
    """
    Compute the combined loss for the VAE with classification: Reconstruction loss + KL divergence + Classification loss.
    
    Arguments:
    recon_x (tensor): The reconstructed input from the decoder.
    x (tensor): The original input data.
    mu (tensor): The mean of the latent variable distribution from the encoder.
    logvar (tensor): The log variance of the latent variable distribution from the encoder.
    classification_output (tensor): The classifier output.
    labels (tensor): The true labels for classification.
    weight_BCE (float): Weight for the reconstruction loss.
    weight_KLD (float): Weight for the KL divergence loss.
    weight_classification_loss (float): Weight for the classification loss.
    
    Returns:
    total_loss (tensor): The combined, weighted loss value (normalized to 1).
    """
    
    # VAE losses
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")  # sum of BCE for all features
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    
    # Classification loss
    classification_loss = F.cross_entropy(classification_output, labels)
    
    # Weighted losses
    BCE_weighted = weight_BCE * BCE
    KLD_weighted = weight_KLD * KLD
    classification_loss_weighted = weight_classification_loss * classification_loss
    
    # Total loss (sum of weighted losses)
    total_loss = BCE_weighted + KLD_weighted + classification_loss_weighted
    
    # Normalize the total loss to be 1 (optional)
    total_loss = total_loss / total_loss.item()  # Normalize by the final loss value
    
    # Return individual losses
    return total_loss, BCE_weighted, KLD_weighted, classification_loss_weighted


def train_VAE_with_classification(model, train_loader, lr=1e-3, num_epochs=10):
    """
    Train a Variational Autoencoder (VAE) with a classification head.
    
    Arguments:
    model: The VAE with classification model.
    train_loader: The training data loader.
    lr: Learning rate.
    num_epochs: Number of epochs for training.
    """
    
    # Automatically detect the device of the model
    device = next(model.parameters()).device
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0  # Initialize total loss for the epoch

        # Iterate over the batches in the training data
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()

            # Move data and labels to the appropriate device
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            encoded, decoded, mu, log_var, classification_output = model(data)

            # Compute the loss with weights for each component
            loss, BCE_weighted, KLD_weighted, classification_loss_weighted = loss_BCE_KLD_classification(
                decoded, data, mu, log_var, classification_output, labels,
                weight_BCE=1.0, weight_KLD=1.0, weight_classification_loss=1.0
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()

            # Print out the individual losses (optional)
            # It's better to print periodically, not on every batch
            if (epoch + 1) % 5 == 0:  # Print every 5 epochs or adjust as needed
                print(f"Batch Losses -> BCE: {BCE_weighted:.4f}, KLD: {KLD_weighted:.4f}, Classification Loss: {classification_loss_weighted:.4f}, Total Loss: {loss:.4f}")
        
        # Compute the average loss for this epoch
        epoch_loss = total_loss / len(train_loader.dataset)  # Total loss / number of examples
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss = {epoch_loss:.4f}")
    
    # Return the trained model
    return model




# def train_VAE(model, train_loader, lr=1e-3, num_epochs=10):
#     """
#     Train a Variational Autoencoder (VAE) model.
    
#     Arguments:
#     model: The VAE model.
#     train_loader: The training data loader.
#     lr: Learning rate.
#     num_epochs: Number of epochs for training.
#     """
    
#     # Automatically detect the device of the model
#     device = next(model.parameters()).device
    
#     # Initialize optimizer
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # Define the loss function
#     criterion = nn.MSELoss()

#     # Training loop
#     for epoch in num_epochs:
        
#         model.train()

#         total_loss = 0.0

#         # Iterate over the batches in the training data
#         for data in tqdm(train_loader, desc="Batch", unit="batch"):
            
#             optimizer.zero_grad()

#             # Get a batch of training data and move it to the device
#             data = data.to(device)

#             # Forward pass
#             encoded, decoded, mu, log_var = model(data)

#             # Compute the loss
#             loss_reconstruction = criterion(decoded, data)
#             loss_kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

#             loss = loss_reconstruction + loss_kld

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             # Update the running total loss
#             total_loss += loss.item() * data.size(0)  # Sum the batch losses

#         # Compute the average loss for this epoch
#         epoch_loss = total_loss / len(train_loader.dataset)
#         print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss:.4f}")


#     # Return the trained model
#     return model
