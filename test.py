import torch
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import SPADEGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Visualization Function
def visualize_output(sar, optical, output, idx):
    sar = sar.cpu().numpy().squeeze()
    optical = optical.cpu().numpy().squeeze().transpose(1, 2, 0)
    output = output.cpu().numpy().squeeze().transpose(1, 2, 0)

    # Denormalize from [-1, 1] to [0, 1]
    optical = (optical + 1) / 2
    output = (output + 1) / 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(sar, cmap='gray')
    ax[0].set_title(f'SAR Image {idx}')
    ax[1].imshow(optical.clip(0, 1))
    ax[1].set_title(f'Optical Image {idx}')
    ax[2].imshow(output.clip(0, 1))
    ax[2].set_title(f'Colorized Output {idx}')
    
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    plt.savefig(f'output_images/output_{idx}.png')
    plt.close(fig)

# Testing Function
def test_model():
    # Parameters
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "sar_colorization_spade.pth"

    # Load test dataset
    test_dataset = SAROpticalDataset('images/sar_test/', 'images/oi_test/')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = SPADEGenerator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Testing Loop
    with torch.no_grad():
        for idx, (sar, optical) in enumerate(test_dataloader):
            sar = sar.float().to(DEVICE)
            optical = optical.float().to(DEVICE)

            # Forward pass
            output = model(sar, sar)

            # Visualize output
            visualize_output(sar[0], optical[0], output[0], idx)

# Run the testing function
test_model()