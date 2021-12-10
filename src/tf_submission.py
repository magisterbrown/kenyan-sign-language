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

