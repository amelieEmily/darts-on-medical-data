from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
###############################################################################

class MalariaImageLabelDataset(Dataset):
    """A dataset class to retrieve samples of paired images and labels"""

    def __init__(self, transform=None):
        """
        Args:
            csv (string): Path to the csv file with data
            shuffle (callable, optional): Shuffle list of files
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        super().__init__()
        FILE_ABSOLUTE_PATH = os.path.abspath(__file__)
        cell_images_folder_path = os.path.dirname(FILE_ABSOLUTE_PATH)
        self.parasitized_path = cell_images_folder_path + '/Parasitized/'
        self.uninfected_path = cell_images_folder_path + '/Uninfected/'
        self.infected = os.listdir(self.parasitized_path)
        self.uninfected = os.listdir(self.uninfected_path)
        self.transform = transform

    def __len__(self):
        return len(self.infected) + len(self.uninfected)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.parasitized_path + self.infected[idx] if idx < len(self.infected) else self.uninfected_path + self.uninfected[idx - len(self.infected)]
        label = 'parasitized' if idx < len(self.infected) else 'uninfected'
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample

