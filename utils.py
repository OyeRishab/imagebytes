import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class SAROpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, img_size=(256, 256)):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.img_size = img_size
        self.sar_images = sorted(os.listdir(sar_dir))
        self.optical_images = sorted(os.listdir(optical_dir))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_img = cv2.imread(os.path.join(self.sar_dir, self.sar_images[idx]), cv2.IMREAD_GRAYSCALE)
        optical_img = cv2.imread(os.path.join(self.optical_dir, self.optical_images[idx]))

        sar_img = cv2.resize(sar_img, self.img_size) / 255.0
        optical_img = cv2.resize(optical_img, self.img_size)

        sar_img = np.expand_dims(sar_img, axis=0)
        optical_img = np.transpose(optical_img, (2, 0, 1))

        return torch.from_numpy(sar_img).float(), torch.from_numpy(optical_img).float()