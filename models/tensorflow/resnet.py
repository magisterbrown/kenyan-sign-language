from classification_models.tfkeras import Classifiers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D

class FullRes34(Model):
  def __init__(self,n_hidden,drop):
    super().__init__()
    ResNet34, preprocess_input = Classifiers.get('resnet34')
    self.back = ResNet34((224, 224, 3),include_top=False)
    self.apool = GlobalAveragePooling2D()
    self.drop = tf.keras.layers.Dropout(drop)
    self.bn = tf.keras.layers.BatchNormalization()
    self.d1 = Dense(n_hidden,activation='relu')
    self.d2 = Dense(9,activation='softmax')

  def load_back(self,path):
    self.back.load_weights(path)

  def set_back(self,weights):
    self.back.set_weights(weights)

  def call(self,x,training):
    x = self.back(x,training=training)
    x = self.apool(x)
    x = self.drop(x,training=training)
    x = self.d1(x)
    x = self.d2(x)
    return x
