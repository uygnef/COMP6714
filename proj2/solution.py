from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import spacy
import pickle
import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE


def process_data(input_data):
    """Extract the first file enclosed in a zip file as a list of words."""
    data = ''
    with zipfile.ZipFile(input_data) as f:
        for i in f.namelist():
            data += tf.compat.as_str(f.read(i)).strip()
    nlp = spacy.load('en')
    paredDate = nlp(data)
    # TODO: do we need lemma??
    ret = []
    for tok in paredDate:
        if tok.lemma_ and not (tok.is_punct or tok.is_space):
            temp = tok.lower_
            if temp[-1] == '%':
                temp = 'PERCENT'
            elif tok.like_num:
                temp = 'NUMBER'
            ret.append(temp)
    file_name = 'tokenize.pickle'
    file = open(file_name, 'wb')
    pickle.dump(ret, file)
    return file_name


def get_probabilty(file_name):
    import math
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    prob_dict = {}
    for i in data:
        if i in prob_dict:
            prob_dict[i] += 1
        else:
            prob_dict[i] = 1
    length = len(data)
    print(length)

    def google_prob(i):
        freq = i / length * 1000
        return min(1, (math.sqrt(freq) + 1) / freq)

    for i in prob_dict:
        prob_dict[i] = google_prob(prob_dict[i])

    with open('prob', 'wb') as prob_file:
        pickle.dump(prob_dict, prob_file)


vocabulary_size = 29677  # This variable is used to define the maximum vocabulary size.


def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    print(count)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print(len(data))
    return data, count, dictionary, reversed_dictionary


data_index = 0


# the variable is abused in this implementation.
# Outside the sample generation loop, it is the position of the sliding window: from data_index to data_index + span
# Inside the sample generation loop, it is the next word to be added to a size-limited buffer.

def generate_batch(batch_size, num_samples, skip_window):
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window

    with open('prob', 'rb') as prob_file:
        prob_dict = pickle.load(prob_file)

    batch = []
    lable = []

    def get_pair(index):
        lable.append(data[index])
        if index <= skip_window:
            begin_index = skip_window
        elif index + skip_window >= len(data):
            begin_index = len(data) - skip_window - 1
        else:
            begin_index = index
        begin_index -= skip_window

        buffer = [i for i in range(begin_index, begin_index + 2 * skip_window + 1) if i != index]
        random.shuffle(buffer)
        for i in buffer:
            if random.random() < prob_dict[reverse_dictionary[data[i]]]:
                batch.append(data[i])
            if len(batch) == num_samples * len(lable):
                break
        while len(batch) < num_samples * len(lable):
            batch.append(buffer[random.randint(0, len(buffer)-1)])

    while len(batch) < batch_size:
        index = random.randint(0, len(data))
        if random.random() < prob_dict[reverse_dictionary[data[index]]]:
            get_pair(index)
    return np.array(batch), np.array(lable)


# file_name = process_data('BBC_Data.zip')
file_name = 'tokenize.pickle'
with open(file_name, 'rb') as file:
    data = pickle.load(file)

get_probabilty(file_name)

data, count, dictionary, reverse_dictionary = build_dataset(data, vocabulary_size)
# Specification of Training data:
batch_size = 128  # Size of mini-batch for skip-gram model.
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right of the target word.
num_samples = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Sample size for negative examples.
logs_path = './log/'

# Specification of test Sample:
sample_size = 20  # Random sample of words to evaluate similarity.
sample_window = 100  # Only pick samples in the head of the distribution.
sample_examples = np.random.choice(sample_window, sample_size, replace=False)  # Randomly pick a sample of size 16

## Constructing the graph...
graph = tf.Graph()

with graph.as_default():
    with tf.device('/cpu:0'):
        # Placeholders to read input data.
        with tf.name_scope('Inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Look up embeddings for inputs.
        with tf.name_scope('Embeddings'):
            sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                                 labels=train_labels, inputs=embed,
                                                 num_sampled=num_sampled, num_classes=vocabulary_size))

        # Construct the Gradient Descent optimizer using a learning rate of 0.01.
        with tf.name_scope('Gradient_Descent'):
            optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Normalize the embeddings to avoid overfitting.
        with tf.name_scope('Normalization'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm

        sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
        similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a summary to monitor cost tensor
        tf.summary.scalar("cost", loss)
        # Merge all summary variables.
        merged_summary_op = tf.summary.merge_all()

num_steps = 130001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    session.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    print('Initializing the model')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op using session.run()
        _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)

        summary_writer.add_summary(summary, step)
        average_loss += loss_val

        if step % 5000 == 0:
            if step > 0:
                average_loss /= 5000

                # The average loss is an estimate of the loss over the last 5000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        # Evaluate similarity after every 10000 iterations.
        if step % 10000 == 0:
            sim = similarity.eval()  #
            for i in range(sample_size):
                sample_word = reverse_dictionary[sample_examples[i]]
                top_k = 10  # Look for top-10 neighbours for words in sample set.
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % sample_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
            print()

    final_embeddings = normalized_embeddings.eval()
