from src.preprocessing.tf_dataset_parsers import TfLabler
from src.preprocessing.tf_dataset_parsers import TfPresenter
import math
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
class CrossBatches(CrossDataset):
  def __init__(self, bs: int, immods, *args,**kwargs):
    super().__init__(*args,**kwargs)
    self.immods = immods
    self.bs = bs

  def get_split(self, test_ids: list):
    train, test = super().get_split(test_ids)
    parser = TfLabler(processors=self.immods)
    train = train.map(parser, num_parallel_calls=tf.data.AUTOTUNE).shuffle(128, reshuffle_each_iteration=True).batch(self.bs,drop_remainder=True)

    presenter = TfPresenter(processors=[self.immods[-1]])
    test = test.map(presenter, num_parallel_calls=tf.data.AUTOTUNE).batch(self.bs)

    return train, test

  def get_sizes(self, test_ids: list):
    trains, tests = super().get_sizes(test_ids)
    trains = trains//self.bs
    tests = int(math.ceil(tests/self.bs))
    return trains, tests