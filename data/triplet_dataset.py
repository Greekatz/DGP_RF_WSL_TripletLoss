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

        # Organize images by class
        self.class_to_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            self.class_to_indices[item['label']].append(idx)

        self.labels = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_item = self.data[idx]
        anchor_img = self._load_image(anchor_item['path'])
        anchor_label = anchor_item['label']

        # Sample a positive from same class (excluding anchor itself)
        pos_indices = self.class_to_indices[anchor_label]
        pos_idx = random.choice([i for i in pos_indices if i != idx])
        positive_item = self.data[pos_idx]
        positive_img = self._load_image(positive_item['path'])

        # Sample a negative from a different class
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        neg_idx = random.choice(self.class_to_indices[neg_label])
        negative_item = self.data[neg_idx]
        negative_img = self._load_image(negative_item['path'])

        return anchor_img, positive_img, negative_img

    def _load_image(self, rel_path):
        full_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
