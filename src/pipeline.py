from preprocessing.dataloader import make_dataloader_from_npy
from config.paths import npy_path

# change the npy path according to which resized image data you would like to ue
dl = make_dataloader_from_npy(npy_path)


