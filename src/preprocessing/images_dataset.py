import numpy as np
from torch.utils.data import Dataset, DataLoader
from config.paths import npy_path, train_path


class ImageDataset(Dataset):
    def __init__(self, data, prediction):
        self.data = data
        self.prediction = prediction
        pass

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        element = self.data[idx]
        prediction = self.prediction[idx]
        return element, prediction


