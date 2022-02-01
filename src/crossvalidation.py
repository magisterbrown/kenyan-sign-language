import tensorflow as tf

class CrossPart:
  counting_batch = 64

  def __init__(self, dataset: tf.data.Dataset):
    self.elements = 0
    self.ds = dataset
    for el in dataset.batch(self.counting_batch):
      self.elements+=el.shape[0]

  def get_ds(self):
    return self.ds

  def __len__(self):
    return self.elements



class CrossDataset:
  def __init__(self, path: str, dataset_names: list):
    self.datasets = list()
    for dsn in dataset_names:
      element = f'{path}/{dsn}'
      dataset = tf.data.TFRecordDataset(element)
      ds = CrossPart(dataset)
      self.datasets.append(ds)

  def get_split(self, test_ids: list):
    train = list()
    test = list()
    for key, ds in enumerate(self.datasets):
      if key in test_ids:
        test.append(ds.get_ds())
      else:
        train.append(ds.get_ds())

    train = CrossDataset.combine_together(train)
    test = CrossDataset.combine_together(test)

    return train, test

  def get_sizes(self, test_ids: list):
    train = 0
    test = 0
    for key, ds in enumerate(self.datasets):
      if key in test_ids:
        test+=len(ds)
      else:
        train+=len(ds)
    return train, test

  @staticmethod
  def combine_together(datasets):
    if len(datasets)>0:
      datasets = tf.data.Dataset.zip(tuple(datasets)).flat_map(
          lambda *args: CrossDataset.concat_datasets(args)
      )
      return datasets

    return None

  @staticmethod
  def concat_datasets(datasets):
    ds0 = tf.data.Dataset.from_tensors(datasets[0])
    for ds1 in datasets[1:]:
        ds0 = ds0.concatenate(tf.data.Dataset.from_tensors(ds1))
        
    return ds0
