import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import ColorizationUNet

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


dataset = SAROpticalDataset('images/SAR1/', 'images/OI1/')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Model, Loss, Optimizer
model = ColorizationUNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for sar, optical in dataloader:
        sar, optical = sar.float().to(DEVICE), optical.float().to(DEVICE)

        # Forward pass
        output = model(sar)
        loss = criterion(output, optical)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "sar_colorization.pth")
