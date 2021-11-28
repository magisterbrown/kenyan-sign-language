from config.paths import train_path
from preprocessing.images_dataset import ImageDataset
from torch.utils.data import DataLoader
from config.paths import resized_npy_path

'''

This script is the overall pipeline for building dataloaders and training the models

!!!!First run the main in convert_to_npy path to save the npy files locally then run this script

'''
if __name__ == '__main__':
    # Create a dataset from the train path and the respective images in the resized_npy_path
    image_ds = ImageDataset(train_path, resized_npy_path)

    # Create the dataloader
    dl = DataLoader(image_ds, batch_size=32, shuffle=True)

    dl_iterator = iter(dl)
    print(next(dl_iterator))
