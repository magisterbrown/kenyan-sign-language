from classification_models.tfkeras import Classifiers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D

class ResNet34(Model):
  def __init__(self):
    super().__init__()
    ResNet34, preprocess_input = Classifiers.get('resnet34')
    self.backbone = ResNet34(input_shape=(480,480,3),weights='imagenet',include_top=False)
    self.pool = GlobalAveragePooling2D()
    self.d1 = Dense(512,activation='relu')
    self.d2 = Dense(12,activation='sigmoid')

  def call(self, x):
    x = self.backbone(x)
    x = self.d1(x)
    x = self.d2(x)
    x = x*25.2-0.1
    x = tf.reshape(x, shape=[-1,1])
    return x
