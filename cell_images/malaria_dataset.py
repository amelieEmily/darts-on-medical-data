from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import os
###############################################################################

class MalariaImageLabelDataset(Dataset):
    """A dataset class to retrieve samples of paired images and labels"""

    def __init__(self, shuffle=None, transform=None):
        """
        Args:
            csv (string): Path to the csv file with data
            shuffle (callable, optional): Shuffle list of files
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        super().__init__()
        self.parasitized_path = '../input/cell_images/cell_images/Parasitized/'
        self.uninfected_path = '../input/cell_images/cell_images/Uninfected/'
        self.infected = os.listdir(self.parasitized_path)
        self.uninfected = os.listdir(self.uninfected_path)
        self.transform = transform

    def __len__(self):
        return len(self.infected) + len(self.uninfected)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.infected + self.infected if idx < len(self.infected) else self.parasitized_path + self.uninfected
        label = 'parasitized' if idx < len(self.infected) else 'uninfected'
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample

###############################################################################
def loadImageToTensor(image_file, transform=None):
    """Load an image and returns a tensor
    Args:
        image_file (string): Path to the image file
        transform (callable, optional): Optional transform to be applied  on a sample.
    """

    image = Image.open(image_file)
    if transform:
        image = transform(image)

    return image.unsqueeze(0)
