import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import SPADEGenerator, Discriminator

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Data
dataset = SAROpticalDataset('images/sar_train/', 'images/oi_train/')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Models
generator = SPADEGenerator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Loss Functions
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LR)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    for sar, optical in dataloader:
        sar = sar.to(DEVICE)
        optical = optical.to(DEVICE)

        # Adversarial ground truths
        valid = torch.ones(sar.size(0), 1, 16, 16).to(DEVICE)
        fake = torch.zeros(sar.size(0), 1, 16, 16).to(DEVICE)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generate images
        gen_optical = generator(sar, sar)

        # Loss measures generator's ability to fool the discriminator
        combined_input = torch.cat((sar, gen_optical), 1)
        pred_fake = discriminator(combined_input)
        loss_GAN = adversarial_loss(pred_fake, valid)
        loss_pixel = pixelwise_loss(gen_optical, optical)
        loss_G = loss_GAN + 100 * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real images
        real_input = torch.cat((sar, optical), 1)
        pred_real = discriminator(real_input)
        loss_real = adversarial_loss(pred_real, valid)

        # Fake images
        fake_input = torch.cat((sar, gen_optical.detach()), 1)
        pred_fake = discriminator(fake_input)
        loss_fake = adversarial_loss(pred_fake, fake)

        # Total loss
        loss_D = (loss_real + loss_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

# Save the generator model
torch.save(generator.state_dict(), "sar_colorization_gan.pth")