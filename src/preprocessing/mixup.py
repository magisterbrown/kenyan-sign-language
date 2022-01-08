import numpy as np
from tensorflow.keras.losses import sparse_categorical_crossentropy

def mixup_loss(y_actual,y_pred):
  l1 = sparse_categorical_crossentropy(y_actual[:,0],y_pred)
  l2 = sparse_categorical_crossentropy(y_actual[:,1],y_pred)
  scalor = y_actual[:,2]
  loss = l1*scalor+l2*(1-scalor)
  return loss

def vec_mix(tensor,scalor,order):
  tensor = tf.cast(tensor,tf.float32)
  mixed_x = tf.gather(tensor,order)
  tensor = tensor*(scalor)+mixed_x*(1-scalor)
  return tensor


def mixer(x,y,maxmix):
  y = tf.cast(y,tf.float32)
  neworder = np.arange(x.shape[0])
  neworder = tf.random.shuffle(neworder)
  scalor = tf.random.uniform([x.shape[0]],0.0,maxmix)
  scalor = tf.reshape(scalor,(x.shape[0],1,1,1))
  x = vec_mix(x,scalor,neworder)
  y_mixed = tf.gather(y,neworder)
  return x, tf.stack((y,y_mixed,tf.squeeze(scalor)), axis=1)

def val_mixer(x,y):
  ones = tf.ones((x.shape[0]), tf.float32)
  y = tf.cast(y,tf.float32)
  return x, tf.stack((y,y,ones), axis=1)
