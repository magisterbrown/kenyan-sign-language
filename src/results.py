import pandas as pd
import numpy as np
import tensorflow as tf

#Generates results file
def generator(loader, model, df):
    count = 0
    for img,idx in loader:
        pred  = model(img,False).numpy()
        for key,row in enumerate(pred):
          idc = idx[key].numpy().decode("utf-8")
          df.loc[idc] = row
        count+=1
        if count%50 == 49:
            print(f'{count}')
    return df

#Predict one fold
def make_df(validation,model):
  reses = list()
  count=0
  for row in validation:
    img,lable,imgname = row
    preds = pd.DataFrame(model(img,training=False ).numpy())
    lable = pd.Series(lable.numpy())
    preds["Lable"] = lable
    preds["id"] = imgname.numpy()
    preds["id"] = preds["id"].apply(lambda x:x.decode("utf-8"))
    reses.append(preds)

  reses = pd.concat(reses)
  return reses

#Cross validate model on the dataset 3-folds
def cross_pred(train_gen,init_model,epochs):
    cpth = f'gs://chimps-first/data/crossval'
    allres = list()
    for i in range(1,4):
      trairec = f'{cpth}/f{i}/train.tfrecords'
      testrec = f'{cpth}/f{i}/test.tfrecords'
      model = init_model()
      trainfold = train_gen()
      history = model.fit(
        trainfold, epochs=epochs,verbose=True
      )
      vparser = TfPresenter(processors=[normalize])
      validation = tf.data.TFRecordDataset(testrec).map(vparser).batch(256)
      testfold = make_df(validation,model)
      allres.append(testfold)
    
    allres = pd.concat(allres)
    return allres
