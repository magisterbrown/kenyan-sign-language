# Path to image directory
import os
from PIL import Image
import os, sys
import numpy as np
from config.paths import image_path, npy_path


def from_image_to_numpy():
    dirs = os.listdir(str(image_path))
    dirs.sort()

    for item in dirs:
        im = Image.open(os.path.join(image_path, item)).convert("RGB")
        if not (os.path.exists(npy_path)):
            os.mkdir(npy_path)
        data_path = os.path.join(npy_path, item[:-4] + '.npy')
        image_array = np.array(im)
        np.save(data_path, image_array)


if __name__ == '__main__':
    from_image_to_numpy()

