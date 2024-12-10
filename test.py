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
    optical = (optical + 1) / 2.0
    output = (output + 1) / 2.0

    # Clip values to [0, 1]
    optical = np.clip(optical, 0, 1)
    output = np.clip(output, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(sar, cmap='gray')
    ax[0].set_title(f'SAR Image {idx}')
    ax[0].axis('off')
    ax[1].imshow(optical)
    ax[1].set_title(f'Optical Image {idx}')
    ax[1].axis('off')
    ax[2].imshow(output)
    ax[2].set_title(f'Colorized Output {idx}')
    ax[2].axis('off')

    # Save the figure
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    plt.savefig(f'output_images/output_{idx}.png', bbox_inches='tight')
    plt.close(fig)