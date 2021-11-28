# Path to image directory
from PIL import Image
import os, sys
import numpy as np
from config.paths import image_path, npy_path, resized_npy_path


def from_image_to_numpy(resize_path=None):
    dirs = os.listdir(str(image_path))
    dirs.sort()
    # npy_path for non resized npys
    path = npy_path

    # resize path for resized image npys
    if resize_path is not None:
        path = resize_path

    for item in dirs:
        image = Image.open(os.path.join(image_path, item)).convert("RGB")

        # resize images to 720x720 to normalize
        if resize_path is not None:
            image = image.resize((720, 720), Image.ANTIALIAS)

        # if path doesn't exist create the directory
        if not (os.path.exists(path)):
            os.mkdir(path)

        # save the image representing npy to the path with the same image name
        data_path = os.path.join(path, item[:-4] + '.npy')
        image_array = np.array(image)
        np.save(data_path, image_array)


if __name__ == '__main__':
    '''
    Run only one 
    1. first one for non resized images
    2. second one for resized images
    '''

    # from_image_to_numpy()
    from_image_to_numpy(resized_npy_path)
