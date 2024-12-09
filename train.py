import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import Generator, Discriminator

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

dataset = SAROpticalDataset('images/sar_train/', 'images/oi_train/')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Models, Loss, and Optimizers
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
criterion_gan = nn.BCELoss()
criterion_pixel = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# Training Loop
for epoch in range(EPOCHS):
    G.train()
    D.train()
    g_total_loss, d_total_loss = 0, 0

    for idx, (sar, optical) in enumerate(dataloader):
        sar, optical = sar.float().to(DEVICE), optical.float().to(DEVICE)

        valid = torch.ones(sar.size(0), 1, 8, 8).to(DEVICE)
        fake = torch.zeros(sar.size(0), 1, 8, 8).to(DEVICE)

        # --- Train Generator ---
        optimizer_G.zero_grad()
        fake_optical = G(sar)
        g_loss_gan = criterion_gan(D(torch.cat((sar, fake_optical), 1)), valid)
        g_loss_pixel = criterion_pixel(fake_optical, optical)
        g_loss = g_loss_gan + 100 * g_loss_pixel
        g_loss.backward()
        optimizer_G.step()

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        real_loss = criterion_gan(D(torch.cat((sar, optical), 1)), valid)
        fake_loss = criterion_gan(D(torch.cat((sar, fake_optical.detach()), 1)), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        g_total_loss += g_loss.item()
        d_total_loss += d_loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], G_Loss: {g_total_loss/len(dataloader):.4f}, D_Loss: {d_total_loss/len(dataloader):.4f}")

torch.save(G.state_dict(), "sar_colorization_g.pth")
torch.save(D.state_dict(), "sar_colorization_d.pth")
