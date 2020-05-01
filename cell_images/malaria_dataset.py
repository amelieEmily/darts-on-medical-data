from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
###############################################################################

class MalariaImageLabelDataset(Dataset):
    """A dataset class to retrieve samples of paired images and labels"""

    def __init__(self, transform, shuffle=None):
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
        included_extensions = ['jpg', 'jpeg', 'png']
        self.infected_path = cell_images_folder_path + '/Parasitized/'
        self.uninfected_path = cell_images_folder_path + '/Uninfected/'
        self.infected_paths  = [self.infected_path + fn for fn in os.listdir(self.infected_path)
              if any(fn.endswith(ext) for ext in included_extensions)]
        self.uninfected_paths = [self.uninfected_path + fn for fn in os.listdir(self.uninfected_path)
              if any(fn.endswith(ext) for ext in included_extensions)]
        self.data = self.infected_paths + self.uninfected_paths
        self.transform = transform
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.infected) + len(self.uninfected)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.data[idx]
        label = 1 if path.startswith(self.infected_path) else 0
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        sample = (image, label)

        return sample