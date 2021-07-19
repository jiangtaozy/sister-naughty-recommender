# Maintained by jemo from 2021.4.12 to now
# Created by jemo on 2021.4.12 17:35:49
# Recommender

from sqlalchemy import create_engine
import pymysql
import pandas as pd
from config import db_connection
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

conn = create_engine(db_connection)

df = pd.read_sql("select userId, imageId, duration from imageBrowseRecord", conn)
#print("df: ", df)

dataset = tf.data.Dataset.from_tensor_slices((df.values))

images = dataset.map(lambda x: x[1])
print("*************images: ", images)
for x in images.take(1).as_numpy_iterator():
  print("**************x: ", x)

#print("**************dataset: ", dataset)

for x in dataset.take(1).as_numpy_iterator():
  print("**************x: ", x)

user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.IntegerLookup()
user_ids_vocabulary.adapt(dataset.map(lambda x: x[0]))

image_ids_vocabulary = tf.keras.layers.experimental.preprocessing.IntegerLookup()
image_ids_vocabulary.adapt(dataset.map(lambda x: x[1]))

print("**************user_ids_vocabulary: ", user_ids_vocabulary)
print("**************image_ids_vocabulary: ", image_ids_vocabulary)

max_duration = dataset.map(lambda x: x[2]).reduce(
  tf.cast(0, tf.int64), tf.maximum
).numpy().max()
min_duration = 0
duration_buckets = np.linspace(
  min_duration,
  max_duration,
  num=1000,
)

class RecommenderModel(tfrs.Model):
  def __init__(self):
    super().__init__()
    self.user_model = tf.keras.Sequential([
      user_ids_vocabulary,
      tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
    ])
    self.duration_model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Discretization(duration_buckets.tolist()),
      tf.keras.layers.Embedding(len(duration_buckets) + 2, 64),
    ])
    self.image_model = tf.keras.Sequential([
      image_ids_vocabulary,
      tf.keras.layers.Embedding(image_ids_vocabulary.vocab_size(), 64)
    ])
    self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
      images.batch(128).map(self.image_model)
    ))
  def compute_loss(self, features, training=False) -> tf.Tensor:
    print("**************features: ", features)
    user_embeddings = self.user_model(features[0])
    duration_embeddings = self.duration_model(features[2])
    image_embeddings = self.image_model(features[1])
    return self.task(user_embeddings, image_embeddings)

model = RecommenderModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
model.fit(dataset.batch(4096), epochs=3)

# scann
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index(images.batch(100).map(model.image_model), images)

_, imageId = scann_index(tf.constant([2]))
print("********************scann index: ", imageId)
