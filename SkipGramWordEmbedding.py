import zipfile
import re
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from adjustText import adjust_text
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K

url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'


def download_data(url, data_dir):
    """Download a file if not present, and make sure it's the right size."""

    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, 'bbc-fulltext.zip')

    if not os.path.exists(file_path):
        print('Downloading file...')
        filename, _ = urlretrieve(url, file_path)
    else:
        print("File already exists")

    extract_path = os.path.join(data_dir, 'bbc')
    if not os.path.exists(extract_path):

        with zipfile.ZipFile(os.path.join(data_dir, 'bbc-fulltext.zip'), 'r') as zipf:
            zipf.extractall(data_dir)

    else:
        print("bbc-fulltext.zip has already been extracted")


download_data(url, 'data')


def read_data(data_dir):
    # This will contain the full list of stories
    news_stories = []

    print("Reading files")

    i = 0  # Just used for printing progress
    for root, dirs, files in os.walk(data_dir):

        for fi, f in enumerate(files):

            # We don't read the readme file
            if 'README' in f:
                continue

            # Printing progress
            i += 1
            print("." * i, f, end='\r')

            # Open the file
            with open(os.path.join(root, f), encoding='latin-1') as f:

                story = []
                # Read all the lines
                for row in f:
                    story.append(row.strip())

                # Create a single string with all the rows in the doc
                story = ' '.join(story)
                # Add that to the list
                news_stories.append(story)

        print('', end='\r')

    print(f"\nDetected {len(news_stories)} stories")
    return news_stories

news_stories = read_data(os.path.join('data', 'bbc'))

# Printing some stats and sample data
print(f"{sum([len(story.split(' ')) for story in news_stories])} words found in the total news set")
print('Example words (start): ',news_stories[0][:50])
print('Example words (end): ',news_stories[-1][-50:])


tokenizer = Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' '
)

tokenizer.fit_on_texts(news_stories)
print("Data fitted on the tokenizer")

n_vocab = len(tokenizer.word_index.items()) + 1

news_sequences = tokenizer.texts_to_sequences(news_stories)
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(n_vocab, sampling_factor=1e-05)


def skip_gram_data_generator(sequences, window_size, batch_size, negative_samples, vocab_size, seed=None):
    rand_sequence_ids = np.arange(len(sequences))
    np.random.shuffle(rand_sequence_ids)

    for si in rand_sequence_ids:

        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequences[si],
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0.0,
            shuffle=False,
            sampling_table=sampling_table,
            seed=seed
        )

        targets, contexts, labels = [], [], []

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=negative_samples,
                unique=True,
                range_max=vocab_size,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            context = tf.concat(
                [tf.constant([context_word], dtype='int64'), negative_sampling_candidates],
                axis=0
            )

            label = tf.constant([1] + [0] * negative_samples, dtype="int64")

            # Append each element from the training example to global lists.
            targets.extend([target_word] * (negative_samples + 1))
            contexts.append(context)
            labels.append(label)

        contexts, targets, labels = np.concatenate(contexts), np.array(targets), np.concatenate(labels)

        assert contexts.shape[0] == targets.shape[0]
        assert contexts.shape[0] == labels.shape[0]

        # If seed is not provided generate a random one
        if not seed:
            seed = random.randint(0, 10e6)

        np.random.seed(seed)
        np.random.shuffle(contexts)
        np.random.seed(seed)
        np.random.shuffle(targets)
        np.random.seed(seed)
        np.random.shuffle(labels)

        for eg_id_start in range(0, contexts.shape[0], batch_size):
            yield (
                targets[eg_id_start: min(eg_id_start + batch_size, targets.shape[0])],
                contexts[eg_id_start: min(eg_id_start + batch_size, contexts.shape[0])]
            ), labels[eg_id_start: min(eg_id_start + batch_size, labels.shape[0])]


batch_size = 4096 # Data points in a single batch

embedding_size = 128 # Dimension of the embedding vector.

window_size=1 # We use a window size of n on either side of target word
negative_samples = 4 # Number of negative samples generated per example

epochs = 5 # Number of epochs to train for

# We pick a random validation set to sample nearest neighbors
valid_size = 16 # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always being deterministic
valid_window = 250

# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well
np.random.seed(54321)
random.seed(54321)

valid_term_ids = np.array(random.sample(range(valid_window), valid_size))
valid_term_ids = np.append(
    valid_term_ids, random.sample(range(1000, 1000+valid_window), valid_size),
    axis=0
)

K.clear_session()

# Inputs - skipgrams() function outputs target, context in that order
# we will use the same order
input_1 = tf.keras.layers.Input(shape=(), name='target')
input_2 = tf.keras.layers.Input(shape=(), name='context')

# Two embeddings layers are used one for the context and one for the target
context_embedding_layer = tf.keras.layers.Embedding(
    input_dim=n_vocab, output_dim=embedding_size, name='context_embedding'
)
target_embedding_layer = tf.keras.layers.Embedding(
    input_dim=n_vocab, output_dim=embedding_size, name='target_embedding'
)

# Lookup outputs of the embedding layers
target_out = target_embedding_layer(input_1)
context_out = context_embedding_layer(input_2)

# Computing the dot product between the two
out = tf.keras.layers.Dot(axes=-1)([context_out, target_out])

# Defining the model
skip_gram_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=out, name='skip_gram_model')

# Compiling the model
skip_gram_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam'
)

skip_gram_model.summary()


class ValidationCallback(tf.keras.callbacks.Callback):

    def __init__(self, valid_term_ids, model_with_embeddings, tokenizer):
        self.valid_term_ids = valid_term_ids
        self.model_with_embeddings = model_with_embeddings
        self.tokenizer = tokenizer

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """ Validation logic """

        # We will use context embeddings to get the most similar words
        # Other strategies include: using target embeddings, mean embeddings after avaraging context/target
        embedding_weights = self.model_with_embeddings.get_layer("context_embedding").get_weights()[0]
        normalized_embeddings = embedding_weights / np.sqrt(np.sum(embedding_weights ** 2, axis=1, keepdims=True))

        # Get the embeddings corresponding to valid_term_ids
        valid_embeddings = normalized_embeddings[self.valid_term_ids, :]

        # Compute the similarity between valid_term_ids and all the embeddings
        # V x d (d x D) => V x D
        top_k = 5  # Top k items will be displayed
        similarity = np.dot(valid_embeddings, normalized_embeddings.T)

        # Invert similarity matrix to negative
        # Ignore the first one because that would be the same word as the probe word
        similarity_top_k = np.argsort(-similarity, axis=1)[:, 1: top_k + 1]

        # Print the output
        for i, term_id in enumerate(valid_term_ids):
            similar_word_str = ', '.join([self.tokenizer.index_word[j] for j in similarity_top_k[i, :] if j >= 1])
            print(f"{self.tokenizer.index_word[term_id]}: {similar_word_str}")

        print('\n')


skipgram_validation_callback = ValidationCallback(valid_term_ids, skip_gram_model, tokenizer)

for ei in range(epochs):
    print(f"Epoch: {ei + 1}/{epochs} started")

    news_skip_gram_gen = skip_gram_data_generator(
        news_sequences, window_size, batch_size, negative_samples, n_vocab
    )

    skip_gram_model.fit(
        news_skip_gram_gen, epochs=1,
        callbacks=skipgram_validation_callback,
    )


def save_embeddings(model, tokenizer, vocab_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    _, words_sorted = zip(*sorted(list(tokenizer.index_word.items()), key=lambda x: x[0])[:vocab_size - 1])

    words_sorted = [None] + list(words_sorted)

    pd.DataFrame(
        model.get_layer("context_embedding").get_weights()[0],
        index=words_sorted
    ).to_pickle(os.path.join(save_dir, "context_embedding.pkl"))

    pd.DataFrame(
        model.get_layer("target_embedding").get_weights()[0],
        index=words_sorted
    ).to_pickle(os.path.join(save_dir, "target_embedding.pkl"))


save_embeddings(skip_gram_model, tokenizer, n_vocab, save_dir='skipgram_embeddings')

skipgram_context_embeddings = pd.read_pickle(
    os.path.join('skipgram_embeddings', 'context_embedding.pkl')
)

skipgram_words, skipgram_embeddings = np.array(skipgram_context_embeddings.index), skipgram_context_embeddings.values


def find_clustered_embeddings(embeddings, top_n=10, top_words=2000):
    '''
    Find only the closely clustered embeddings.
    This gets rid of more sparsly distributed word embeddings and make the visualization clearer
    This is useful for t-SNE visualization

    distance_threshold: maximum distance between two points to qualify as neighbors
    sample_threshold: number of neighbors required to be considered a cluster
    '''

    # calculate cosine similarity
    embeddings_norm = embeddings / np.sum(embeddings ** 2, keepdims=True, axis=1)
    cosine_sim = np.dot(embeddings, embeddings.T)

    sim_sum_words = np.sum(np.sort(cosine_sim, axis=1)[:, -top_n:], axis=1)

    clustered_words = np.argsort(-sim_sum_words)[:top_words]

    return clustered_words

import sklearn


tsne = TSNE(
    perplexity=30, n_components=2, n_iter=5000, metric='cosine')

print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold

# prune the embeddings by getting ones only more than n-many sample above the similarity threshold
# this unclutters the visualization
plot_word_ids = find_clustered_embeddings(
    skipgram_embeddings, top_n=25, top_words=300
)


skipgram_embeddings_norm = skipgram_embeddings / np.sum(skipgram_embeddings**2, keepdims=True, axis=1)
tsne_embeddings = tsne.fit_transform(skipgram_embeddings_norm)

print('Pruning the T-SNE embeddings')

print('\tDone')


def plot(embeddings, labels):
    print(f"Plotting {embeddings.shape[0]} points")
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(15, 15))  # in inches

    # plot all the embeddings and their corresponding words
    plot_annotations = []

    for i, label in enumerate(labels):

        x, y = embeddings[i, :]
        plt.scatter(x, y, alpha=0.25, c='gray')

        # We only annotate a small fraction of data, to avoid long waiting times
        if np.random.normal() > 0.75:
            plot_annotations.append(
                plt.text(x, y, label, ha='center', va='center', fontsize=10)
            )

    print(
        f"Adjusting text annotations in the plot. This may take some time: Annotations {len(plot_annotations)}"
    )
    adjust_text(plot_annotations)

    # use for saving the figure if needed
    # plt.savefig('word_embeddings.png')
    plt.show()


plot(tsne_embeddings[1:1501], skipgram_words[1:1501])