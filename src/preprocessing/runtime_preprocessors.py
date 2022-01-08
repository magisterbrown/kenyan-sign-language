import tensorflow as tf

def resize(image,side=224):
  image = tf.image.resize(image,(224,224))
  return image

def normalize(image,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  image = image/255
  image = image - mean
  image = image/std
  return image
