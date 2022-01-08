#Create tf effeceint net V2 model 
class EffiecntHead(Model):
  def __init__(self,n_hidden,drop,back):
    super().__init__()
    self.back = back
    self.drop = tf.keras.layers.Dropout(drop)
    self.d1 = Dense(n_hidden,activation='relu')
    self.d2 = Dense(9,activation='softmax')

  def load_back(self,path):
    self.back.load_weights(path)

  def set_back(self,weights):
    self.back.set_weights(weights)

  def call(self,x,training):
    x = self.back(x,training=training)
    x = self.drop(x,training=training)
    x = self.d1(x)
    x = self.d2(x)
    
    return x
