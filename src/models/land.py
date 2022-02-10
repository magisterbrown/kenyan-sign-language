import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import PReLU

#Create cnn with head from landmark-recognition-2020 competition winners
class LandmarkHead(Model):
  def __init__(self,back):
    super().__init__()
    self.back = back
    self.head = tf.keras.Sequential([
            Dense(512),
            BatchNormalization(),
            PReLU(name='embedding'),
    ])
    self.final = tf.keras.Sequential([ Dense(9,activation='softmax')])

  def save_point(self,path: str):
    self.back.save(f'{path}/back.h5')
    self.head.save(f'{path}/head.h5')
    self.final.save(f'{path}/final.h5')

  def load_point(self,path: str):
    self.back.load_weights(f'{path}/back.h5')
    self.head.load_weights(f'{path}/head.h5')
    self.final.load_weights(f'{path}/final.h5')

  def load_back(self,path):
    self.back.load_weights(path)

  def get_embed(self,x):
    x = self.back(x,training=False)
    embed = self.head(x,training=False)
    pred = self.final(embed,training=False)
    
    return embed,pred

  def call(self,x,training):
    x = self.back(x,training=training)
    x = self.head(x,training=training)
    x = self.final(x,training=training)
    
    return x
    