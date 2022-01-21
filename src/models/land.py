import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

#Create cnn with head from landmark-recognition-2020 competition winners
class LandmarkHead(Model):
  def __init__(self,back):
    super().__init__()
    self.back = back
    self.d1 = Dense(512)
    self.activation = PReLU()
    self.bn = BatchNormalization()
    self.d2 = Dense(9,activation='softmax')

  def load_back(self,path):
    self.back.load_weights(path)

  def set_back(self,weights):
    self.back.set_weights(weights)

  def call(self,x,training):
    x = self.back(x,training=training)
    x = self.d1(x,training=training)
    x = self.bn(x,training=training)
    x = self.activation(x,training=training)
    x = self.d2(x,training=training)
