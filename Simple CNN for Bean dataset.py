# import necessary libraries 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# load beans dataset
(ds_train, ds_test), info = tfds.load('beans', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info = True)

# method to normalize and resize image
# (images are originally shaped (500, 500, 3), which take a super long time to train)
def normalize_and_resize_img(image, label):
    return tf.cast(tf.image.resize(image, (64, 64)), tf.float32) / 255., label

# training and testing sets set up
ds_train = ds_train.map(normalize_and_resize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(info.splits["train"].num_examples)
ds_train = ds_train.batch(64)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_and_resize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_train.batch(64)
ds_test = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Simple CNN model, Input -> Conv(32) -> Pooling -> Conv(64) -> Pooling -> Conv(128) -> FC(64) -> FC(10)
model = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)

# Use Adam optimizer
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics = ["accuracy"],
)

model.fit(ds_train, epochs = 50, verbose = 2)
model.evaluate(ds_test)