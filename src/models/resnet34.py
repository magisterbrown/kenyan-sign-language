from classification_models.keras import Classifiers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D

class FullRes34(Model):
  def __init__(self):
    super().__init__()
    ResNet34, preprocess_input = Classifiers.get('resnet34')
    self.back = ResNet34((224, 224, 3),include_top=False)
    self.pool = GlobalAveragePooling2D()
    self.d1 = Dense(512,activation='relu')
    self.d2 = Dense(9,activation='softmax')

  def load_back(self,path):
    self.back.load_weights(path)
  def __call__(self,x,training):
    x = self.back(x,training=training)
    x = self.pool(x)
    x = self.d1(x)
    x = self.d2(x)
    return x
