import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os

''' 
data_path: path to the train csv
image_array_path: path to the saved image npys
'''
class ImageDataset(Dataset):
    def __init__(self, data_path, image_array_path):
        train_df = pd.read_csv(data_path)
        self.data = []
        self.prediction = []

        # Load the npy file, add to data and add the append the respective label from the dataframe
        for idx, row in train_df.iterrows():
            id = row['img_IDS']
            img_path = os.path.join(image_array_path, id + '.npy')
            if os.path.exists(img_path):
                img = (np.load(img_path))
                self.data.append(img)
                self.prediction.append(row['Label'])

        self.data = np.array(self.data)
        pass

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        element = self.data[idx]
        prediction = self.prediction[idx]
        return element, prediction
