import tensorflow as tf

#Parses image from tf.dataset
class TfParser:
    def __init__(self,processors=list()):
      self.processors = processors
      self.features = {
            'image': tf.io.FixedLenFeature([], tf.string)
            }
    
    def __call__(self, example):
        content = tf.io.parse_single_example(example, self.features)
        output = self.construct_output(content)
        return output

    def construct_output(self, content):
        image = self.process_image(content)
        return image

    def process_image(self, content):
        image = tf.io.parse_tensor(content["image"],tf.uint8)
        image = tf.reshape(image, shape=[480,480,3])
        for operation in self.processors:
          image = operation(image)
        return image
#Parses image with lable int from tf.dataset
class TfLabler(TfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features['lable'] = tf.io.FixedLenFeature([], tf.int64)

    def construct_output(self, content):
        lable = tf.cast(content["lable"],tf.int32)
        image = self.process_image(content)
        return image, lable

#Parse images for the submission file
class TfSubmiter(TfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features['id'] = tf.io.FixedLenFeature([], tf.string)

    def construct_output(self, content):
        lable = content["id"]
        image = self.process_image(content)
        return image, lable

#Parse datasets to create cross validation predictions
class TfPresenter(TfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features['lable'] = tf.io.FixedLenFeature([], tf.int64)
        self.features['id'] = tf.io.FixedLenFeature([], tf.string)

    def construct_output(self, content):
        lable = tf.cast(content["lable"],tf.int32)
        id = content["id"]
        image = self.process_image(content)
        return image, lable, id

