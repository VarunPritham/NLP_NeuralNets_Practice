import collections
import math
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

seed = 54321

url = 'https://github.com/ZihanWangKi/CrossWeigh/raw/master/data/'
dir_name = 'data'


def download_data(url, filename, download_dir, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""

    # Create directories if doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # If file doesn't exist download
    if not os.path.exists(os.path.join(download_dir, filename)):
        filepath, _ = urlretrieve(url + filename, os.path.join(download_dir, filename))
    else:
        filepath = os.path.join(download_dir, filename)

    # Check the file size
    statinfo = os.stat(filepath)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filepath)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filepath + '. Can you get to it with a browser?')

    return filepath


# Filepaths to train/valid/test data
train_filepath = download_data(url, 'conllpp_train.txt', dir_name, 3283420)
dev_filepath = download_data(url, 'conllpp_dev.txt', dir_name, 827443)
test_filepath = download_data(url, 'conllpp_test.txt', dir_name, 748737)


def read_data(filename):
    '''
    Read data from a file with given filename
    Returns a list of sentences (each sentence a string),
    and list of ner labels for each string
    '''

    print("Reading data ...")
    # master lists - Holds sentences (list of tokens), ner_labels (for each token an NER label)
    sentences, ner_labels = [], []

    # Open the file
    with open(filename, 'r', encoding='latin-1') as f:
        # Read each line
        is_sos = True  # We record at each line if we are seeing the beginning of a sentence

        # Tokens and labels of a single sentence, flushed when encountered a new one
        sentence_tokens = []
        sentence_labels = []
        i = 0
        for row in f:
            # If we are seeing an empty line or -DOCSTART- that's a new line
            if len(row.strip()) == 0 or row.split(' ')[0] == '-DOCSTART-':
                is_sos = False
            # Otherwise keep capturing tokens and labels
            else:
                is_sos = True
                token, _, _, ner_label = row.split(' ')
                sentence_tokens.append(token)
                sentence_labels.append(ner_label.strip())

            # When we reach the end / or reach the beginning of next
            # add the data to the master lists, flush the temporary one
            if not is_sos and len(sentence_tokens) > 0:
                sentences.append(' '.join(sentence_tokens))
                ner_labels.append(sentence_labels)
                sentence_tokens, sentence_labels = [], []

    print('\tDone')
    return sentences, ner_labels


# Train data
train_sentences, train_labels = read_data(train_filepath)
# Validation data
valid_sentences, valid_labels = read_data(dev_filepath)
# Test data
test_sentences, test_labels = read_data(test_filepath)


from itertools import chain

# Print the value count for each label
print("Training data label counts")
print(pd.Series(chain(*train_labels)).value_counts())

print("\nValidation data label counts")
print(pd.Series(chain(*valid_labels)).value_counts())

print("\nTest data label counts")
print(pd.Series(chain(*test_labels)).value_counts())

print("Train Sentences Length Distribution",pd.Series(train_sentences).str.split().str.len().describe(percentiles=[0.05, 0.95]))

def get_label_id_map(train_labels):
    # Get the unique list of labels
    unique_train_labels = pd.Series(chain(*train_labels)).unique()
    # Create a class label -> class ID mapping
    labels_map = dict(zip(unique_train_labels, np.arange(unique_train_labels.shape[0])))
    print(f"labels_map: {labels_map}")
    return labels_map

labels_map = get_label_id_map(train_labels)

def get_padded_int_labels(labels, labels_map, max_seq_length, return_mask=True):
    # Convert string labels to integers
    int_labels = [[labels_map[x] for x in one_seq] for one_seq in labels]

    # Pad sequences
    if return_mask:
        # If we return mask, we first pad with a special value (-1) and
        # use that to create the mask and later replace -1 with 'O'
        padded_labels = np.array(
            tf.keras.preprocessing.sequence.pad_sequences(
                int_labels, maxlen=max_seq_length, padding='post', truncating='post', value=-1
            )
        )

        # mask filter
        mask_filter = (padded_labels != -1)
        # replace -1 with 'O' s ID
        padded_labels[~mask_filter] = labels_map['O']
        return padded_labels, mask_filter.astype('int')

    else:
        padded_labels = np.array(ner_pad_sequence_func(int_labels, value=labels_map['O']))
        return padded_labels


# The maximum length of sequences
max_seq_length = 40

# Size of token embeddings
embedding_size = 64

# Number of hidden units in the RNN layer
rnn_hidden_size = 64

# Number of output nodes in the last layer
n_classes = 9

# Number of samples in a batch
batch_size = 64

# Number of epochs to train
epochs = 3


# Convert string labels to integers for all train/validation/test data
# Pad train/validation/test data
padded_train_labels, train_mask = get_padded_int_labels(
    train_labels, labels_map, max_seq_length, return_mask=True
)
padded_valid_labels, valid_mask = get_padded_int_labels(
    valid_labels, labels_map, max_seq_length, return_mask=True
)
padded_test_labels, test_mask  = get_padded_int_labels(
    test_labels, labels_map, max_seq_length, return_mask=True
)

K.clear_session()


def get_fitted_token_vectorization_layer(corpus, max_seq_length, vocabulary_size=None):
    """ Fit a TextVectorization layer on given data """

    # Define a text vectorization layer
    vectorization_layer = TextVectorization(
        max_tokens=vocabulary_size, standardize=None,
        output_sequence_length=max_seq_length,
    )
    # Fit it on a corpus of data
    vectorization_layer.adapt(corpus)

    # Get the vocabulary size
    n_vocab = len(vectorization_layer.get_vocabulary())

    return vectorization_layer, n_vocab

# Input layer
word_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)

# Text vectorize layer
vectorize_layer, n_vocab = get_fitted_token_vectorization_layer(train_sentences, max_seq_length)

# Vectorized output (each word mapped to an int ID)
vectorized_out = vectorize_layer(word_input)

# Look up embeddings for the returned IDs
embedding_layer = layers.Embedding(input_dim=n_vocab, output_dim=embedding_size, mask_zero=True)(vectorized_out)

# Define a simple RNN layer, it returns an output at each position
rnn_layer = layers.SimpleRNN(
    units=rnn_hidden_size, return_sequences=True
)

rnn_out = rnn_layer(embedding_layer)

dense_layer = layers.Dense(n_classes, activation='softmax')
dense_out = dense_layer(rnn_out)

model = tf.keras.Model(inputs=word_input, outputs=dense_out)


def macro_accuracy(y_true, y_pred):
    # [batch size * time]
    y_true = tf.cast(tf.reshape(y_true, [-1]), 'int32')
    y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), 'int32')

    sorted_y_true = tf.sort(y_true)
    sorted_inds = tf.argsort(y_true)

    sorted_y_pred = tf.gather(y_pred, sorted_inds)

    sorted_correct = tf.cast(tf.math.equal(sorted_y_true, sorted_y_pred), 'int32')

    # We are adding one to make sure ther eare no division by zero
    correct_for_each_label = tf.cast(tf.math.segment_sum(sorted_correct, sorted_y_true), 'float32') + 1
    all_for_each_label = tf.cast(tf.math.segment_sum(tf.ones_like(sorted_y_true), sorted_y_true), 'float32') + 1

    mean_accuracy = tf.reduce_mean(correct_for_each_label / all_for_each_label)

    return mean_accuracy

mean_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=macro_accuracy, name='macro_accuracy')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[mean_accuracy_metric])

model.summary()


def get_class_weights(train_labels):
    label_count_ser = pd.Series(chain(*train_labels)).value_counts()
    label_count_ser = label_count_ser.sum() / label_count_ser
    label_count_ser /= label_count_ser.max()

    label_id_map = get_label_id_map(train_labels)
    label_count_ser.index = label_count_ser.index.map(label_id_map)
    return label_count_ser.to_dict()


def get_sample_weights_from_class_weights(labels, class_weights):
    """ From the class weights generate sample weights """
    return np.vectorize(class_weights.get)(labels)


train_class_weights = get_class_weights(train_labels)

# Make train_sequences an array
train_sentences = np.array(train_sentences)
# Get sample weights (we cannot use class_weight with TextVectorization layer)
train_sample_weights = get_sample_weights_from_class_weights(padded_train_labels, train_class_weights)

# Training the model
model.fit(
        train_sentences, padded_train_labels,
        sample_weight=train_sample_weights,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(np.array(valid_sentences), padded_valid_labels)
)

n_samples = 5
visual_test_sentences = test_sentences[:n_samples]
visual_test_labels = padded_test_labels[:n_samples]

visual_test_predictions = model.predict(np.array(visual_test_sentences))
visual_test_pred_labels = np.argmax(visual_test_predictions, axis=-1)

rev_labels_map = dict(zip(labels_map.values(), labels_map.keys()))
for i, (sentence, sent_labels, sent_preds) in enumerate(zip(visual_test_sentences, visual_test_labels, visual_test_pred_labels)):
    n_tokens = len(sentence.split())
    print("Sample:\t","\t".join(sentence.split()))
    print("True:\t","\t".join([rev_labels_map[i] for i in sent_labels[:n_tokens]]))
    print("Pred:\t","\t".join([rev_labels_map[i] for i in sent_preds[:n_tokens]]))
    print("\n")