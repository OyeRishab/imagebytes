import torch
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import SPADEGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to save generated images
def save_generated_image(sar, gen_optical, idx):
    sar = sar.cpu().numpy().squeeze()
    gen_optical = gen_optical.cpu().numpy().squeeze().transpose(1, 2, 0)

    # Denormalize from [-1, 1] to [0, 1]
    gen_optical = (gen_optical + 1) / 2.0

    # Clip values to [0, 1]
    gen_optical = np.clip(gen_optical, 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sar, cmap='gray')
    ax[0].set_title(f'SAR Image {idx}')
    ax[0].axis('off')
    ax[1].imshow(gen_optical)
    ax[1].set_title(f'Generated Optical Image {idx}')
    ax[1].axis('off')

    # Save the figure
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    plt.savefig(f'generated_images/generated_{idx}.png', bbox_inches='tight')
    plt.close(fig)

# Testing Function
def test_model():
    # Parameters
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "sar_colorization_gan.pth"  # Path to the trained GAN generator model

    # Load test dataset (SAR images only)
    test_dataset = SAROpticalDataset('images/sar_test/', 'images/oi_test/')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Generator
    generator = SPADEGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(MODEL_PATH))
    generator.eval()

    # Generate new images
    with torch.no_grad():
        for idx, (sar, _) in enumerate(test_dataloader):
            sar = sar.to(DEVICE)
            # Generate images
            gen_optical = generator(sar, sar)
            # Save generated images
            batch_size = sar.size(0)
            for i in range(batch_size):
                save_generated_image(sar[i], gen_optical[i], idx * BATCH_SIZE + i)

# Run the testing function
test_model()