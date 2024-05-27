# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tqdm import tqdm

# %%
track_uri_to_track_name = {}
track_uri_to_artist_name = {}
track_uri_to_album_name = {}

corpus = []


try:
    corpus = open('corpus.txt', 'r').read()
    track_uri_to_track_name = json.load(open('track_uri_to_track_name.json', 'r'))
    track_uri_to_artist_name = json.load(open('track_uri_to_artist_name.json', 'r'))
    track_uri_to_album_name = json.load(open('track_uri_to_album_name.json', 'r'))
except:
    file_count = 0
    for file in tqdm(os.listdir('data/raw_data')):
        #! THIS IS DONE SO AS TO SAVE ON COMPUTATION TIME
        file_count += 1
        if file_count > 200:
            break

        json_file = open(f'data/raw_data/{file}', 'r')
        data = json.load(json_file)
        _playlists = data['playlists']

        for playlist in _playlists:
            temp_playlist = []

            for song in playlist['tracks']:
                track_uri = song['track_uri']

                track_name = song['track_name']
                artist_name = song['artist_name']
                album_name = song['album_name']

                if track_uri not in track_uri_to_track_name:
                    track_uri_to_track_name[track_uri] = track_name
                    track_uri_to_artist_name[track_uri] = artist_name
                    track_uri_to_album_name[track_uri] = album_name

                temp_playlist.append(track_uri)

            corpus.append(' '.join(temp_playlist))
    corpus = '\n'.join(corpus)

    
    with open('corpus.txt', 'w') as f:
        f.write(corpus)

    with open('track_uri_to_track_name.json', 'w') as f:
        json.dump(track_uri_to_track_name, f)
    with open('track_uri_to_artist_name.json', 'w') as f:
        json.dump(track_uri_to_artist_name, f)
    with open('track_uri_to_album_name.json', 'w') as f:
        json.dump(track_uri_to_album_name, f)

# TIME: 6 minutes

# %%
# https://www.tensorflow.org/text/tutorials/word2vec

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# %% [markdown]
# # Data prep

# %%
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

# %%
text_ds = tf.data.TextLineDataset('./corpus.txt')
# .filter(lambda x: tf.cast(tf.strings.length(x), bool)) 
#! This is not needed a s there are no empty lines (playlists) in the corpus

def custom_standardization(input_data):
  return input_data
  #! this too does not need to be changed as the data is already in the correct format

# Define the vocabulary size and the number of words in a sequence.
vocab_size = 100000 #! THIS LIMITS THE NUMBER OF SONGS TO THE MOST COMMON 100,000
sequence_length = 10 #! THIS STANDARDIZES THE LENGTH OF DATA WITH PADDING

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(text_ds.batch(1024))

# %%
# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])

# %%
# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))   

# %%
for seq in sequences[:5]:
    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

# %%
window_size = 3
num_ns = 4

try:
    targets = np.load('targets.npy')
    contexts = np.load('contexts.npy')
    labels = np.load('labels.npy')
except:
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=window_size,
        num_ns=num_ns,
        vocab_size=vocab_size,
        seed=SEED)

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    # %%
    # save the dataset to disk
    np.save('targets.npy', targets)
    np.save('contexts.npy', contexts)
    np.save('labels.npy', labels)

targets_val = targets[int(len(targets) * 0.8):]
contexts_val = contexts[int(len(contexts) * 0.8):]
labels_val = labels[int(len(labels) * 0.8):]

targets = targets[:int(len(targets) * 0.8)]
contexts = contexts[:int(len(contexts) * 0.8)]
labels = labels[:int(len(labels) * 0.8)]

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")
print('\n')
print(f"targets_val.shape: {targets_val.shape}")
print(f"contexts_val.shape: {contexts_val.shape}")
print(f"labels_val.shape: {labels_val.shape}")
# %%
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
# Create a training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# Create a validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices(((targets_val, contexts_val), labels_val))
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

# %%
class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots
    
def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

# %%
embedding_dim = 100
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(
                     from_logits=True),
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
word2vec.fit(train_dataset, epochs=12, validation_data=val_dataset, callbacks=[tensorboard_callback])

# %%
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# %%
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()


#  save the vocab and weights to disk
np.save('vocab.npy', vocab)
np.save('weights.npy', weights)
