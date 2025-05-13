import os
import random
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class TripletDataset(Dataset):
    def __init__(self, data_list, root_dir, transform=None):
        """
        Args:
            data_list (List[Dict]): [{'path': str, 'label': int}, ...]
            root_dir (str): Base directory for image paths
            transform (callable): Transform to apply to images (e.g., torchvision transforms)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        anchor = self._load_image(row["anchor"])
        positive = self._load_image(row["positive"])
        negative = self._load_image(row["negative"])

        return anchor, positive, negative


    def _load_image(self, rel_path):
        full_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
