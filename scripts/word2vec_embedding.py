# Embedding

import io
import os
import re
import shutil
import string
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Defining vocab size, sequence length, embedding dimension, batch size and epochs
vocab_size = 10000
sequence_length = 20
embedding_dim = 16
batch_size = 10
epochs = 25

## creating the dataset

os.chdir('/Users/josea/Desktop/')

seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'corpus_2/train', batch_size=batch_size, validation_split=0.2, 
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'corpus_2/train', batch_size=batch_size, validation_split=0.2, 
    subset='validation', seed=seed)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Embed a 1000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# define the custom standartization that put the text
# in lowercase and remove the punctuation
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '') 


# Use the text vectorization layer to normalize, split, and map strings to 
# integers. Note that the layer uses the custom standardization defined above. 
# Set maximum_sequence length as all samples are not of the same length.

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# Create the model
model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit the model
history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=epochs)


# save the loss and accuracy for both test and train datasets
left_data_1 = history.history['accuracy']
left_data_2 = history.history['val_accuracy']
right_data_1 = history.history['loss']
right_data_2 = history.history['val_loss']

with open("accuracy__TRAIN_seqsize_{}_embdim_{}_vocabsize_{}_batch_{}_epoch_{}.txt".format(sequence_length, embedding_dim, vocab_size, batch_size, epochs), "w") as output:
    output.write(str(left_data_1))
        
with open("loss_TRAIN_seqsize_{}_embdim_{}_vocabsize_{}_batch_{}_epoch_{}.txt".format(sequence_length, embedding_dim, vocab_size, batch_size, epochs), "w") as output:
    output.write(str(right_data_1))
    
with open("accuracy_TEST_seqsize_{}_embdim_{}_vocabsize_{}_batch_{}_epoch_{}.txt".format(sequence_length, embedding_dim, vocab_size, batch_size, epochs), "w") as output:
    output.write(str(left_data_2))
    
with open("loss_TEST_seqsize_{}_embdim_{}_vocabsize_{}_batch_{}_epoch_{}.txt".format(sequence_length, embedding_dim, vocab_size, batch_size, epochs), "w") as output:
    output.write(str(right_data_2))

