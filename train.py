import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import SPADEGenerator

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load Data
dataset = SAROpticalDataset('images/sar_train/', 'images/oi_train/')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Number of training samples: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

# Initialize Model, Loss, Optimizer
model = SPADEGenerator().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for idx, (sar, optical) in enumerate(dataloader):
        sar, optical = sar.float().to(DEVICE), optical.float().to(DEVICE)

        # Print filenames
        sar_filename = dataset.sar_images[idx]
        optical_filename = dataset.optical_images[idx]
        print(f"SAR Image: {sar_filename}, Optical Image: {optical_filename}")

        # Forward pass
        output = model(sar, sar)  # SAR image used as segmentation map
        loss = criterion(output, optical)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "sar_colorization_spade.pth")