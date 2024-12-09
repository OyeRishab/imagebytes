import torch
from torch.utils.data import DataLoader
from utils import SAROpticalDataset
from model import ColorizationUNet
import matplotlib.pyplot as plt
import numpy as np

def visualize_output(sar, output, idx):
    sar = sar.cpu().numpy().squeeze()
    output = output.cpu().numpy().squeeze().transpose(1, 2, 0)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sar, cmap='gray')
    ax[0].set_title(f'SAR Image {idx}')
    ax[1].imshow(output)
    ax[1].set_title(f'Colorized Output {idx}')
    plt.show()

def test_model():
    # Parameters
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "sar_colorization.pth"

    # Load test dataset
    test_dataset = SAROpticalDataset('images/sar_test/', 'images/oi_test/')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = ColorizationUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Testing Loop
    with torch.no_grad():
        for idx, (sar, optical) in enumerate(test_dataloader):
            sar = sar.float().to(DEVICE)
            optical = optical.float().to(DEVICE)

            # Forward pass
            output = model(sar)

            # Visualize output
            visualize_output(sar[0], output[0], idx)

test_model()