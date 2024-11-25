

# Load h5ad

# Exctract matrix

# Put to dense

# Split in train and test

# Create dataloader
train_loader = torch.utils.data.DataLoader(
    X_train, batch_size=batch_size, shuffle=True
)
# one entity is one cells with its genes values

# Instantiate model
model = AutoEncoder()
model.to(device)

