import numpy as np
import os
import pandas as pd

from config.paths import train_path
from src.preprocessing.images_dataset import ImageDataset
from torch.utils.data import DataLoader


def make_dataloader_from_npy(npy_path):
    train_df = pd.read_csv(train_path)
    data = []
    labels = []

    for idx, row in train_df.iterrows():
        id = row['img_IDS']
        print(os.path.join(npy_path, id + '.npy'))
        if os.path.exists(os.path.join(npy_path, id + '.npy')):
            img = (np.load(os.path.join(npy_path, id + '.npy')))
            data.append(img)
            labels.append(row['Label'])

    data = np.array(data)
    image_ds = ImageDataset(data, labels)
    print(len(image_ds))
    dl = DataLoader(image_ds, batch_size=32, shuffle=True)
    return dl
