import pandas as pd
import tensorflow as tf
def generator(loader, model, df):
    count = 0
    for img,idx in loader:
        img = tf.expand_dims(img, axis=0)
        pred  = model(img,False).numpy()[0]
        idx = idx.numpy().decode("utf-8")
        df.loc[idx] = pred
        count+=1
        if count%50 == 49:
            print(f'{count}/{df.shape[0]}')

    
    return df

# Cross validation predictions

import math

class CrossBatches(CrossDataset):
  def __init__(self, bs: int, immods, *args,**kwargs):
    super().__init__(*args,**kwargs)
    self.immods = immods
    self.bs = bs

  def get_split(self, test_ids: list):
    train, test = super().get_split(test_ids)
    parser = TfLabler(processors=self.immods)
    train = train.map(parser, num_parallel_calls=tf.data.AUTOTUNE).shuffle(128, reshuffle_each_iteration=True).batch(self.bs,drop_remainder=True)

    presenter = TfPresenter(processors=self.immods[-1])
    test = test.map(presenter, num_parallel_calls=tf.data.AUTOTUNE).batch(self.bs)

    return train, test

  def get_sizes(self, test_ids: list):
    trains, tests = super().get_sizes(test_ids)
    trains = trains//self.bs
    tests = int(math.ceil(tests/self.bs))
    return trains, tests


class CrossTrain:
  def __init__(self, dataset: CrossDataset, model):
    self.model = model
    model.save_point("crossstart")
    self.ds = dataset

  def case(self, train):
    self.model.load_point("crossstart")
    self.model.fit(train, epochs=27, verbose=False)

  def make_pred(self,test):
    embeddings = list()
    predictions = list()
    for el in test:
      image,lable,id = el
      embed,pred = model.get_embed(image)
      ids = pd.Series(id.numpy())
      lables = pd.Series(lable.numpy())

      embeddings.append(self.res_to_df(embed,ids,lables))
      predictions.append(self.res_to_df(pred,ids,lables))

    embeddings = pd.concat(embeddings)
    predictions = pd.concat(predictions)
    return embeddings, predictions

  @staticmethod
  def res_to_df(res,ids,lables):
    df = pd.DataFrame(res.numpy())
    df["lable"] = lables
    df["id"] = ids
    return df

  def test_combinations(self):
    return np.reshape(np.arange(10,),(-1,2))

  def train(self):

    all_empbs = list()
    all_preds = list()

    for key, comb in enumerate(self.test_combinations()):
      train,test = self.ds.get_split(comb)
      print(f'Model: {key+1}')
      self.case(train)
      emb,prd = self.make_pred(test)
      all_empbs.append(emb)
      all_preds.append(prd)

    all_empbs = pd.concat(all_empbs)
    all_preds = pd.concat(all_preds)

    return all_empbs, all_preds
