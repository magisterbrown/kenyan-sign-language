from preprocessing.dataloader import make_dataloader_from_npy
from config.paths import npy_path
from preprocessing.convert_to_npy import from_image_to_numpy

'''
This script is the overall pipeline for building dataloaders and training the models
'''

# If you want to resize and resave the images uncomment this part
# make resize field True and add a new path to save the npy files
# from_image_to_numpy()

# Change the npy path according to which resized image data you would like to ue
dl = make_dataloader_from_npy(npy_path)


