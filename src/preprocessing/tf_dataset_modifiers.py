import tensorflow as tf

#Parses image from tf.dataset
class TfParser:
    def __init__(self):
        self.fetures = {
            'image': tf.io.FixedLenFeature([], tf.string)
            }
    
    def __call__(self, example):
        content = tf.io.parse_single_example(example, self.features)
        output = self.construct_output(content)
        return output

    def construct_output(self, content):
        image = self.process_image(image)
        return image

    def process_image(self, content):
        image = tf.io.parse_tensor(content["image"],tf.uint8)
        image = tf.reshape(image, shape=[480,480,3])
        return image

#Parses image with lable vector from tf.dataset
class TfLabler(TfParser):
    def __init__(self):
        super().__init__()
        self.fetures['lable'] = tf.io.FixedLenFeature([], tf.int64)

    def construct_output(self, content):
        lable = tf.cast(content["lable"],tf.int32)
        image = self.process_image(image)
        return image, lable
