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
    
    def loss_BCE_KLD(recon_x, x, mu, logvar):
        # Compute the binary cross-entropy loss between the reconstructed output and the input data
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        # Compute the Kullback-Leibler divergence between the learned latent variable distribution and a standard Gaussian distribution
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Combine the two losses by adding them together and return the result
        return BCE + KLD

def train_VAE(model, train_loader, lr, num_epochs=10):
    """
    Train a Variational Autoencoder (VAE) model.
    
    Arguments:
    model: The VAE model.
    train_loader: The training data loader.
    lr: Learning rate.
    num_epochs: Number of epochs for training.
    """
    
    # Automatically detect the device of the model
    device = next(model.parameters()).device
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the loss function
    criterion = nn.MSELoss()

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        
        model.train()

        total_loss = 0.0

        # Iterate over the batches in the training data
        for data in tqdm(train_loader, desc="Batch", unit="batch"):
            optimizer.zero_grad()

            # Get a batch of training data and move it to the device
            data = data.to(device)

            # Forward pass
            encoded, decoded = model(data)

            # Compute the loss
            loss = criterion(decoded, data)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running total loss
            total_loss += loss.item() * data.size(0)  # Sum the batch losses

        # Compute the average loss for this epoch
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss:.4f}")

