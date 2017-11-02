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
from six.moves import xrange

import numpy as np
from six.moves import urllib
import tensorflow as tf
from six.moves import range

import collections
import pickle

data_index = 0


# def generate_batch(data, batch_size, skip_window, reverse_dictionary, prob_dict):
#     """
#     Generates a mini-batch of training data for the training CBOW
#     embedding model.
#     :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
#         training corpus, with words encoded as an integer
#     :param batch_size (int): size of the batch to generate
#     :param skip_window (int): number of words to both left and right that form
#         the context window for the target word.
#     Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch
#     containing all the context words, with the corresponding label being the word in the middle of the context
#     """
#     global data_index
#     assert batch_size > 0
#     # batch = np.ndarray(shape=(batch_size, skip_window*2), dtype=np.int32)
#     labels = []
#     batch = []
#     # span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#     # buffer = collections.deque(maxlen=span)
#     # for _ in range(span):
#     #     buffer.append(data[data_index])
#     #     data_index = (data_index + 1) % len(data)
#
#     for i in range(batch_size):
#         # context tokens are just all the tokens in buffer except the target
#         inc = 0
#         temp = []
#         while len(temp) < 2 * skip_window:
#             for _ in [-1, 1]:
#                 sample_index = data_index + _ * inc
#                 if sample_index < 0:
#                     sample_index = 0
#                 elif sample_index > len(data) - 1:
#                     sample_index = len(data) - 1
#                 if reverse_dictionary[data[sample_index]] == 'UNK':
#                     temp.append(data[sample_index])
#                 elif random.random() < prob_dict[data[sample_index]]:
#                     temp.append(data[sample_index])
#                 # try:
#                 #     if random.random() < prob_dict[reverse_dictionary[data[sample_index]]]:
#                 #         temp.append(data[sample_index])
#                 # except KeyError:
#                 #     print(sample_index)
#                 #     print(data[sample_index])
#                 #     print(reverse_dictionary[data[sample_index]])
#                 #     print(prob_dict)
#                 #     print(prob_dict[reverse_dictionary[data[sample_index]]])
#                 if len(temp) == 2 * skip_window:
#                     break
#             inc += 1
#         batch.append(temp)
#         # batch[i, :] = [token for idx, token in enumerate(buffer) if idx != skip_window]
#         labels.append(data[data_index])
#         # buffer.append(data[data_index])
#         data_index += 1
#         data_index = (data_index + 1) % len(data)
#     return np.array(batch), np.array([labels]).T

def generate_batch(data, batch_size, skip_window, rev = None, dic= None):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch
    containing all the context words, with the corresponding label being the word in the middle of the context
    """
    global data_index
    assert batch_size > 0
    batch = np.ndarray(shape=(batch_size, skip_window * 2), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size):
        # context tokens are just all the tokens in buffer except the target
        batch[i, :] = [token for idx, token in enumerate(buffer) if idx != skip_window]
        labels[i, 0] = buffer[skip_window]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    context_embeds = tf.nn.embedding_lookup(embeddings, train_inputs)
    return tf.reduce_mean(context_embeds, 1)


def process_data(input_data):
    """
    :param input_data: zip file name
    :return: pickle file name
    """
    """Extract the first file enclosed in a zip file as a list of words."""
    import os.path
    file_name = 'tokenize.pickle'

    if os.path.isfile(file_name):
        return file_name
    data = ''
    with zipfile.ZipFile(input_data) as f:
        for i in f.namelist():
            print(i)
            data += tf.compat.as_str(f.read(i)).strip()
    nlp = spacy.load('en')
    paredDate = nlp(data)
    ret = []
    for tok in paredDate:
        if tok.lemma_ and not (tok.is_punct or tok.is_space):
            if tok.ent_type_:
                ret.append(tok.ent_type_)
            elif tok.like_num:
                ret.append('NUM')
            elif tok.lower_[-1] == '%':
                ret.append('PERCENT')
            else:
                ret.append(tok.lower_)
    with open(file_name, 'wb') as file:
        pickle.dump(ret, file)
    return file_name


def get_probabilty(data):
    import math
    prob_dict = {}
    for i in data:
        if i in prob_dict:
            prob_dict[i] += 1
        else:
            prob_dict[i] = 1
    length = len(data)

    def google_prob(i):
        freq = i / length * 100
        return min(1, (math.sqrt(freq) + 1) / freq)

    for i in prob_dict:
        prob_dict[i] = google_prob(prob_dict[i])
    return prob_dict


def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(max(1000, n_words - 200)))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    print(dictionary)
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# print(1)
# file_name = process_data('./BBC_Data.zip')
# print(2)
# # prob_dict = get_probabilty(data)
# print(3)
# # data, count, dictionary, reverse_dictionary = build_dataset(data, vocabulary_size)
# print(4)
# # batch, labels = generate_batch(data, batch_size=128, skip_window=2)
# print(5)
# # Step 4: Build and train a CBOW model.
# print("data", data)
# print("count", count)
# print("dict", dictionary)
# print("rev", reverse_dictionary)
# print("len_rev", len(reverse_dictionary.keys()))
# # exit(0)
# batch_size = 128
# embedding_size = 200  # Dimension of the embedding vector.
# skip_window = 2  # How many words to consider left and right.
#
# # We pick a random validation set to sample nearest neighbors. Here we limit the
# # validation samples to the words that have a low numeric ID, which by
# # construction are also the most frequent.

#
# graph = tf.Graph()
#
# with graph.as_default():
#     # Input data.
#     train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2 * skip_window])
#     train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#     valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
#
#     # Ops and variables pinned to the CPU because of missing GPU implementation
#     with tf.device('/cpu:0'):
#         # Look up embeddings for inputs.
#         embeddings = tf.Variable(
#             tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#
#         # train_inputs is of shape (batch_size, 2*skip_window)
#
#         # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]
#
#         mean_context_embeds = \
#             get_mean_context_embeds(embeddings, train_inputs)
#
#         # Construct the variables for the NCE loss
#         nce_weights = tf.Variable(
#             tf.truncated_normal([vocabulary_size, embedding_size],
#                                 stddev=1.0 / math.sqrt(embedding_size)))
#         nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
#
#         # Compute the average NCE loss for the batch.
#         # tf.nce_loss automatically draws a new sample of the negative labels each
#         # time we evaluate the loss.
#         loss = tf.reduce_mean(
#             tf.nn.sampled_softmax_loss(weights=nce_weights,
#                                        biases=nce_biases,
#                                        labels=train_labels,
#                                        inputs=mean_context_embeds,
#                                        num_sampled=num_sampled,
#                                        num_classes=vocabulary_size))
#
#         # Construct the SGD optimizer using a learning rate of 1.0.
#         optimizer = tf.train.AdamOptimizer().minimize(loss)
#
#         # Compute the cosine similarity between minibatch examples and all embeddings.
#         norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
#         normalized_embeddings = embeddings / norm
#         valid_embeddings = tf.nn.embedding_lookup(
#             normalized_embeddings, valid_dataset)
#         similarity = tf.matmul(
#             valid_embeddings, normalized_embeddings, transpose_b=True)
#
#     # Add variable initializer.
#     init = tf.global_variables_initializer()
#
# # Step 5: Begin training.
# num_steps = 1000
#
# saver = tf.train.Saver({'embedding': embeddings}, max_to_keep=None)
#
# with tf.Session(graph=graph) as session:
#     # We must initialize all variables before we use them.
#     init.run()
#     print('Initialized')
#
#     average_loss = 0
#     for step in xrange(num_steps):
#         batch_inputs, batch_labels = generate_batch(data, batch_size, skip_window)
#         feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
#
#         # We perform one update step by evaluating the optimizer op (including it
#         # in the list of returned values for session.run()
#         _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#         average_loss += loss_val
#
#         if step % 2000 == 0:
#             if step > 0:
#                 average_loss /= 2000
#             # The average loss is an estimate of the loss over the last 2000 batches.
#             print('Average loss at step ', step, ': ', average_loss)
#             average_loss = 0
#
#         # Note that this is expensive (~20% slowdown if computed every 500 steps)
#         if step % 10000 == 0:
#             sim = similarity.eval()
#             for i in range(valid_size):
#                 valid_word = reverse_dictionary[valid_examples[i]]
#                 top_k = 8  # number of nearest neighbors
#                 nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#                 log_str = 'Nearest to %s:' % valid_word
#                 for k in range(top_k):
#                     close_word = reverse_dictionary[nearest[k]]
#                     log_str = '%s %s,' % (log_str, close_word)
#                 print(log_str)
#             np.save("CBOW_Embeddings", normalized_embeddings.eval())
#             # saver.save(session, 'w2vEmbedding', global_step=step)
#
#     final_embeddings = normalized_embeddings.eval()
#     np.save("CBOW_Embeddings", final_embeddings)
#
# with open("all_res.ickle", 'wb') as file:
#     pickle.dump((data, count, dictionary, reverse_dictionary, final_embeddings), file)
#
# file = open("word_embed", 'w')

def adjective_embeddings(data_file, embedding_file_name, num_steps, embedding_dim):
    with open(data_file, 'rb') as file:
        processed_data = pickle.load(file)

    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 10  # Number of negative examples to sample.
    batch_size = 128
    skip_window = 2  # How many words to consider left and right.

    data, count, dictionary, reverse_dictionary = build_dataset(processed_data, 20782)
    vocabulary_size = len(reverse_dictionary.keys())  # This variable is used to define the maximum vocabulary size.
    print(vocabulary_size)
    prob_dict = get_probabilty(data)

    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2 * skip_window])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))

            # train_inputs is of shape (batch_size, 2*skip_window)

            # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]

            mean_context_embeds = \
                get_mean_context_embeds(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_dim],
                                    stddev=1.0 / math.sqrt(embedding_dim)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=nce_weights,
                                           biases=nce_biases,
                                           labels=train_labels,
                                           inputs=mean_context_embeds,
                                           num_sampled=num_sampled,
                                           num_classes=vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.AdamOptimizer().minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    # Step 5: Begin training.

    saver = tf.train.Saver({'embedding': embeddings}, max_to_keep=None)

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, skip_window, reverse_dictionary, prob_dict)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                np.save("CBOW_Embeddings", normalized_embeddings.eval())
                saver.save(session, 'w2vEmbedding', global_step=step)

        final_embeddings = normalized_embeddings.eval()

    with open(embedding_file_name, 'w') as file:
        file.writelines(str(final_embeddings.shape[0]) + ' ' + str(final_embeddings.shape[1]) + '\n')
        for i, value in enumerate(final_embeddings):
            file.write(reverse_dictionary[i])
            for j in value:
                file.writelines(' ' + str(round(j, 6)))
            file.writelines('\n')


def Compute_topk(model_file, input_adjective, top_k):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    result = word_vectors.most_similar(input_adjective, [], top_k)
    return result

out_put = '10k_100_cbow'
file_name = process_data('./BBC_Data.zip')
adjective_embeddings(file_name, out_put, 0, 100)
import shutil
shutil.copyfile('word2vec_fns.py', out_put+".py")
