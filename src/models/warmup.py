from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import tensorflow as tf

class WarmupSchedule(LearningRateSchedule):

  def __init__(self, warmup_steps: int, original: LearningRateSchedule):
    self.warmup_steps = warmup_steps
    self.original = original
    self.target_lr =  original.initial_learning_rate

  @tf.function
  def __call__(self, step: int) -> float:
    if step<self.warmup_steps:
      part = (step+1)/self.warmup_steps
      lr = self.target_lr*part
    else:
      decay_step = step-self.warmup_steps
      lr = self.original(decay_step)

    return lr
