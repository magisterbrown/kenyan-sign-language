import tensorflow as tf
def normalize(image,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  image = tf.image.resize(image,(224,224))
  image = image/255
  image = image - mean
  image = image/std
  return image
