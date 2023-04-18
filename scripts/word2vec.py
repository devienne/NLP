## Word2vec_2

import io
import itertools
import numpy as np
import string
import os
import re
import tensorflow as tf
import tqdm
import pandas
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 42
print(tf.__version__)
get_ipython().run_line_magic('load_ext', 'tensorboard')

######################################################################################

######################################################################################

# create an unique file to append the content of 50 publications in the corpus folder

path = '/Users/josea/Desktop/test'
files = os.listdir(path)

txt_1 = open(os.path.join(path, 'test_2.txt'), 'a+')

for file in files[0:50]:
    txt_2 = open(os.path.join(path, file), 'r')
    txt_1.write('\n')
    txt_1.write('\n')
    txt_1.write(txt_2.read())
    txt_2.close()
    
txt_1.close()

# Use this file to create the text dataset to be analysed with Word2Vec
FILE = '/Users/josea/Desktop/test/test_2.txt'
text_ds = tf.data.TextLineDataset(FILE).filter(lambda x: tf.cast(tf.strings.length(x), bool))


######################################################################################

######################################################################################


# create a custom standardization function to lowercase the text and remove punctuation.

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

# Define the vocabulary size and number of words in a sequence.

vocab_size = 4096 ############################# HHHHHHEEEEEEEEERRRRRRRRRREEEEEEEEEEEEEEEE ####################
sequence_length = 30 ############################# HHHHHHEEEEEEEEERRRRRRRRRREEEEEEEEEEEEEEEE ####################

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)


# call 'adapt' on the text dataset to create a vocabulary
vectorize_layer.adapt(text_ds.batch(1024))


# save the created vocabulary for reference
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])


# the vectorize_layer can now be used to generate
# vectors from each element in text_ds
def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return tf.squeeze(vectorize_layer(text))

# vectorize the data in text_ds
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()


# Put all in one single function
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence, 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=SEED, 
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


targets, contexts, labels = generate_training_data(
    sequences=sequences, # HERE
    window_size=8, # HERE
    num_ns=16, 
    vocab_size=vocab_size, # HERE
    seed=SEED)
print(len(targets), len(contexts), len(labels))


BATCH_SIZE = 100
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)



# Model and training
class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

# define loss function
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


# build up the model
num_ns = 4
embedding_dim = 128 # HERE
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# callback to log training statistics for tensorboard
import datetime
log_dir = "/Users/josea/Desktop/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# training the model
history = word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])


# plot accuracy and loss function
fig = plt.figure()
ax = fig.add_subplot(111)
ln1 = ax.plot(left_data, label = 'Accuracy', color = 'red')
ax2 = ax.twinx()
ln2 = ax2.plot(right_data, label = 'Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
ax.tick_params(axis = 'x', direction = 'in')
ax2.tick_params(axis = 'y', direction = 'in')
ax.tick_params(axis = 'y', direction = 'in')

ln = ln1+ln2
labs = [l.get_label() for l in ln]
ax.legend(ln, labs, loc=0, facecolor="white")

fig.savefig('loss_accuracy', dpi = 300)

# get the weights (vectors) and the vocabulary of the embedded layer

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()


# Create and save the vectors and metadata file.
out_v = io.open('skipgram_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('skipgram_metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if  index == 0: continue # skip 0, it's padding.
  vec = weights[index] 
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()


try:
  from google.colab import files
  files.download('skipgram_vectors.tsv')
  files.download('skipgram_metadata.tsv')
except Exception as e:
  pass

