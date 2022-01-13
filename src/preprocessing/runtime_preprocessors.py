import tensorflow as tf
import sys
sys.path.append("../../models/tensorflow/automl/efficientnetv2")
from effnetv2_model import get_model
from autoaugment import _parse_policy_info
from hparams import Config
import tensorflow.compat.v1 as tf1

def resize(image,side=224):
  image = tf.image.resize(image,(224,224))
  return image

def normalize(image,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  image = image/255
  image = image - mean
  image = image/std
  return image

def randaugment_with_chioce(image, num_layers, magnitude, available_ops):

  replace_value = [128] * 3
  augmentation_params = Config(cutout_const=40, translate_const=100)

  for layer_num in range(num_layers):
    op_to_select = tf1.random_uniform(
        [], maxval=len(available_ops), dtype=tf1.int32)
    random_magnitude = float(magnitude)
    with tf1.name_scope('randaug_layer_{}'.format(layer_num)):
      for (i, op_name) in enumerate(available_ops):
        prob = tf1.random_uniform([], minval=0.2, maxval=0.8, dtype=tf1.float32)
        func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                           replace_value, augmentation_params)
        image = tf1.cond(
            tf1.equal(i, op_to_select),
            lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args),
            lambda: image)
  return image 
