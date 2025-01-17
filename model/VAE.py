import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import multiprocessing as mp

from utils import *

"""
Contains Models Arcchitecture.
"""


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
    batch_size=64):

    # Automatically detect the device of the model
    device = next(model.parameters()).device

    # Save true adn predicted labels
    true_labels = []
    predicted_labels = []

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
            _, _, _, _, logits_classification = model(x)

            # Apply Softmax to logits to get probabilities
            probabilities = torch.softmax(logits_classification, dim=1)

            # Predict the class with the highest probability
            batch_predicted_labels = torch.argmax(probabilities, dim=1)

            # Store predictions
            predicted_labels.extend(batch_predicted_labels.cpu().numpy().tolist())  # Convert to list of ints
            true_labels.extend(y.cpu().numpy().tolist())  # Convert to list of ints

    return true_labels, predicted_labels


def train_VAE_with_classification(
    model, 
    adata,
    fold_indices, 
    lr=1e-5, 
    num_epochs=5,
    batch_size=64,
    weigth_losses = [1, 0.01, 1], #reconstruction, kl, classification
    patience_early_stopping = 10,
    num_workers=None  # Number of parallel workers
):
    
    # Extarct weigth of losses
    w_rec, w_kl, w_cls = weigth_losses

    # Automatically detect the device of the model
    device = next(model.parameters()).device
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Indices of cells in train and val
    indices_train = fold_indices[0] # ATTENTION: the indices are in random order, not in increasing order
    indices_val = fold_indices[1]
    num_samples_train = len(indices_train)
    num_samples_val = len(indices_val)
    print(f"num_samples_train: {num_samples_train}, num_samples_val: {num_samples_val}")

    # Save lists
    epoch_losses_train = []
    epoch_losses_val = []

    # Instantiate early stopping --> for epochs
    early_stopping = EarlyStopping(max_patience=patience_early_stopping)

    # Training loop
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0  # Initialize total loss for the epoch

        t1 = time.perf_counter()

        # Create list with batch indices
        batch_indices_train = [indices_train[i:i + batch_size] for i in range(0, len(indices_train), batch_size)]
        num_batches_train = len(batch_indices_train)
        print(f"\nnum_batches_train: {num_batches_train}")

        t2 = time.perf_counter()
        print("\tCalculate indicies", t2-t1)

        # Iterate over the batches in the training data
        for single_batch_indices_train in tqdm(batch_indices_train, desc=f"Train Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            
            optimizer.zero_grad()

            t1 = time.perf_counter()

            # Ectract data and labels of this batch
            adata_tmp = adata[single_batch_indices_train, ]
            y = adata_tmp.obs["concat_label_encoded"].values.tolist()
            x = adata_tmp.X.toarray() #num_cells(batch_size) x num_genes

            t2 = time.perf_counter()
            print("\tSlice AnnData", t2-t1)

            t1 = time.perf_counter()

            # Move data and labels to the appropriate device
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            # print(f"x shape: {x.shape}, y shape: {y.shape}")
            # print("sum gene expression for each cell")
            # print(torch.sum(x, dim=1))

            t2 = time.perf_counter()
            print("\tTransfor data to tensor", t2-t1)

            t1 = time.perf_counter()

            # Forward pass
            encoded, decoded, mu, log_var, logits_classification = model(x)

            t2 = time.perf_counter()
            print("\tForward pass", t2-t1)

            # Print mu and log_var with detailed formatting
            # print(f"Mu: {mu.detach().cpu().numpy() if mu.is_cuda else mu.detach().numpy()} | "
            #     f"\nLog Variance (log_var): {log_var.detach().cpu().numpy() if log_var.is_cuda else log_var.detach().numpy()}")

            # Compute the loss with weights for each component
            loss, BCE_weighted, KLD_weighted, classification_loss_weighted = losses_VAE(
                decoded, x, mu, log_var, logits_classification, y,
                weight_reconstruction=w_rec, weight_KLD=w_kl, weight_classification_loss=w_cls
            )

            t1 = time.perf_counter()

            # Backward pass
            loss.backward()
            optimizer.step()

            t2 = time.perf_counter()
            print("\tBackpropagation and Step", t2-t1)

            # Accumulate the total loss
            total_loss += loss.item()

            t1 = time.perf_counter()

            del x, y, adata_tmp
            torch.cuda.empty_cache()

            t2 = time.perf_counter()
            print("\tDelete Elements", t2-t1)

        
        # Compute the average loss for this epoch
        epoch_loss_train = total_loss / num_samples_train # Total loss / number of examples
        epoch_losses_train.append(epoch_loss_train)        

        #####################
        # Evaluation

        t1 = time.perf_counter()

        with torch.no_grad():

            model.eval()
            total_loss = 0.0  # Initialize total loss for the epoch

            # Create list with batch indices
            batch_indices_val = [indices_val[i:i + batch_size] for i in range(0, len(indices_val), batch_size)]
            num_batches_val = len(batch_indices_val)

            # Iterate over the batches in the training data
            for single_batch_indices_val in tqdm(batch_indices_val, desc=f"Val Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                
                # Ectract data and labels of this batch
                adata_tmp = adata[single_batch_indices_val, ]
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
            epoch_loss_val = total_loss / num_samples_val # Total loss / number of examples
            epoch_losses_val.append(epoch_loss_val)  

            print(f"\nEpoch ended, Train Loss: {epoch_loss_train:.2f}, Val Loss: {epoch_loss_val:.2f}")

        t2 = time.perf_counter()
        print("\tEvaluate on Val", t2-t1)

        # At the edn of each epogh call the erly stoopping
        early_stopping(epoch_loss_val, model)
        if early_stopping.early_stop: # Check if early stopping has been triggered
            print("Early stopping triggered, Loading previous model.")
            model.load_state_dict(early_stopping.best_model_weights)
            break
    

    # Return the trained model
    return model, epoch_losses_train, epoch_losses_val

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