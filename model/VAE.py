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
        
        # mu and log_var are simpleNN that will meant the mean and std from the latent space emb
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
    weight_classification_loss=1.0
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
    
    # Normalize the total loss to be 1 (optional)
    #total_loss = total_loss / total_loss.item()  # Normalize by the final loss value
    
    print(f"Total Loss: {total_loss.item():.4f} | "
        f"Reconstruction Loss (weighted): {rec_loss_weighted.item():.4f} | "
        f"KLD (weighted): {KLD_weighted.item():.4f} | "
        f"Classification Loss (weighted): {classification_loss_weighted.item():.4f}")
    
    return total_loss, rec_loss_weighted, KLD_weighted, classification_loss_weighted


def train_VAE_with_classification(
    model, 
    adata,
    fold_indices, 
    lr=1e-5, 
    num_epochs=5,
    batch_size=64,
    weigth_losses = [1, 0.0001, 1] #reconstruction, kl, classification
):
    
    # Extarct weigth of losses
    w_rec = weigth_losses[0]
    w_kl = weigth_losses[1]
    w_cls = weigth_losses[2]

    # Automatically detect the device of the model
    device = next(model.parameters()).device
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Indices of cells in train and test
    indices_train = fold_indices[0] # ATTENTION: the indices are in random order, not in increasing order
    indices_test = fold_indices[1]
    num_samples_train = len(indices_train)
    num_samples_test = len(indices_test)
    print(f"num_samples_train: {num_samples_train}, num_samples_test: {num_samples_test}")

    # Save lists
    epoch_losses_train = []
    epoch_losses_test = []

    # Training loop
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0  # Initialize total loss for the epoch

        # Create list with batch indices
        batch_indices_train = [indices_train[i:i + batch_size] for i in range(0, len(indices_train), batch_size)]
        num_batches_train = len(batch_indices_train)
        print(f"\nnum_batches_train: {num_batches_train}")

        # Iterate over the batches in the training data
        for single_batch_indices_train in tqdm(batch_indices_train, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            
            optimizer.zero_grad()

            # Ectract data and labels of this batch
            adata_tmp = adata[single_batch_indices_train, ]
            y = adata_tmp.obs["concat_label_encoded"].values.tolist()
            x = adata_tmp.X.toarray() #num_cells(batch_size) x num_genes

            # Move data and labels to the appropriate device
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            print(f"x shape: {x.shape}, y shape: {y.shape}")

            # Forward pass
            encoded, decoded, mu, log_var, logits_classification = model(x)

            # Print mu and log_var with detailed formatting
            print(f"Mu: {mu.detach().cpu().numpy() if mu.is_cuda else mu.detach().numpy()} | "
                f"\nLog Variance (log_var): {log_var.detach().cpu().numpy() if log_var.is_cuda else log_var.detach().numpy()}")


            # Compute the loss with weights for each component
            loss, BCE_weighted, KLD_weighted, classification_loss_weighted = losses_VAE(
                decoded, x, mu, log_var, logits_classification, y,
                weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()

            del x, y, adata_tmp
            torch.cuda.empty_cache()
        
        # Compute the average loss for this epoch
        epoch_loss_train = total_loss / num_samples_train # Total loss / number of examples
        epoch_losses_train.append(epoch_loss_train)        

        #####################
        # Evaluation

        with torch.no_grad():

            model.eval()
            total_loss = 0.0  # Initialize total loss for the epoch

            # Create list with batch indices
            batch_indices_test = [indices_test[i:i + batch_size] for i in range(0, len(indices_test), batch_size)]
            num_batches_test = len(batch_indices_test)

            # Iterate over the batches in the training data
            for single_batch_indices_test in tqdm(batch_indices_test, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                
                # Ectract data and labels of this batch
                adata_tmp = adata[single_batch_indices_test, ]
                y = adata_tmp.obs["concat_label_encoded"].values.tolist()
                x = adata_tmp.X.toarray() #num_cells(batch_size) x num_genes

                # Move data and labels to the appropriate device
                x = torch.tensor(x, dtype=torch.float32).to(device)
                y = torch.tensor(y, dtype=torch.long).to(device)

                # Forward pass
                encoded, decoded, mu, log_var, logits_classification = model(x)

                # Compute the loss with weights for each component
                loss, BCE_weighted, KLD_weighted, classification_loss_weighted = losses_VAE(
                    decoded, x, mu, log_var, logits_classification, y,
                    weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
                )

                # Accumulate the total loss
                total_loss += loss.item()

                del x, y, adata_tmp
                torch.cuda.empty_cache()
            
            # Compute the average loss for this epoch
            epoch_loss_test = total_loss / num_samples_test # Total loss / number of examples
            epoch_losses_test.append(epoch_loss_test)  

            print(f"Train Loss: {epoch_loss_train:.2f}, Test Loss: {epoch_loss_test:.2f}")


    # Return the trained model
    return model, epoch_losses_train, epoch_losses_test



# TODO: Conditional VAE
# class ConditionalVAE(VAE):
#     # VAE implementation from the article linked above
#     def __init__(self, num_classes):
#         super().__init__()
#         # Add a linear layer for the class label
#         self.label_projector = nn.Sequential(
#             nn.Linear(num_classes, self.num_hidden),
#             nn.ReLU(),
#         )

#     def condition_on_label(self, z, y):
#         projected_label = self.label_projector(y.float())
#         return z + projected_label

#     def forward(self, x, y):
#         # Pass the input through the encoder
#         encoded = self.encoder(x)
#         # Compute the mean and log variance vectors
#         mu = self.mu(encoded)
#         log_var = self.log_var(encoded)
#         # Reparameterize the latent variable
#         z = self.reparameterize(mu, log_var)
#         # Pass the latent variable through the decoder
#         decoded = self.decoder(self.condition_on_label(z, y))
#         # Return the encoded output, decoded output, mean, and log variance
#         return encoded, decoded, mu, log_var

#     def sample(self, num_samples, y):
#         with torch.no_grad():
#             # Generate random noise
#             z = torch.randn(num_samples, self.num_hidden).to(device)
#             # Pass the noise through the decoder to generate samples
#             samples = self.decoder(self.condition_on_label(z, y))
#         # Return the generated samples
#         return samples