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


def cbow_grams(sequence, vocabulary_size,
               window_size=4, negative_samples=1., shuffle=True,
               categorical=False, sampling_table=None, seed=None):
    targets, contexts, labels = [], [], []

    for i, wi in enumerate(sequence):

        if not wi or i < window_size or i + 1 > len(sequence) - window_size:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)

        context_words = [wj for j, wj in enumerate(sequence[window_start:window_end]) if j + window_start != i]
        target_word = wi

        context_classes = tf.expand_dims(tf.constant(context_words, dtype="int64"), 0)

        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_classes,
            num_true=window_size * 2,
            num_sampled=negative_samples,
            unique=True,
            range_max=vocabulary_size,
            name="negative_sampling")

        # Build context and label vectors (for one target word)
        negative_targets = negative_sampling_candidates.numpy().tolist()

        target = [target_word] + negative_targets
        label = [1] + [0] * negative_samples

        # Append each element from the training example to global lists.
        targets.extend(target)
        contexts.extend([context_words] * (negative_samples + 1))
        labels.extend(label)

    couples = list(zip(targets, contexts))

    seed = random.randint(0, 10e6)
    random.seed(seed)
    random.shuffle(couples)
    random.seed(seed)
    random.shuffle(labels)

    return couples, labels

window_size = 1 # How many words to consider left and right.

inputs, labels = tf.keras.preprocessing.sequence.skipgrams(
    tokenizer.texts_to_sequences([news_stories[0][:150]])[0],
    vocabulary_size=len(tokenizer.word_index.items())+1,
    window_size=window_size, negative_samples=4, shuffle=False,
    categorical=False, sampling_table=None, seed=None
)

i = 0
for inp, lbl in zip(inputs, labels):
    i += 1
    print(f"Input: {inp} ({[tokenizer.index_word[wi] for wi in inp]}) / Label: {lbl}")

batch_size = 4096 # Data points in a single batch

embedding_size = 128 # Dimension of the embedding vector.

window_size=1 # We use a window size of 1 on either side of target word
epochs = 5 # Number of epochs to train for
negative_samples = 4 # Number of negative samples generated per example

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


# Inputs; target input layer will have the final shape [None]
# context will have [None, 2xwindow_size] shape
input_1 = tf.keras.layers.Input(shape=())
input_2 = tf.keras.layers.Input(shape=(window_size*2,))

# Target and context embedding layers
target_embedding_layer = tf.keras.layers.Embedding(
    input_dim=n_vocab, output_dim=embedding_size, name='target_embedding'
)

context_embedding_layer = tf.keras.layers.Embedding(
    input_dim=n_vocab, output_dim=embedding_size, name='context_embedding'
)

# Outputs of the target and context embedding lookups
context_out = context_embedding_layer(input_2)
target_out = target_embedding_layer(input_1)

# Taking the mean over the all the context words to produce [None, embedding_size]
mean_context_out = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(context_out)

# Computing the dot product between the two
out = tf.keras.layers.Dot(axes=-1)([context_out, target_out])

cbow_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=out, name='cbow_model')

cbow_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam'
)

cbow_model.summary()

def cbow_data_generator(sequences, window_size, batch_size, negative_samples):
    rand_sequence_ids = np.arange(len(sequences))
    np.random.shuffle(rand_sequence_ids)

    for si in rand_sequence_ids:
        inputs, labels = cbow_grams(
            sequences[si],
            vocabulary_size=n_vocab,
            window_size=window_size,
            negative_samples=negative_samples,
            shuffle=True,
            sampling_table=sampling_table,
            seed=None
        )

        inputs_context, inputs_target, labels = np.array([inp[1] for inp in inputs]), np.array(
            [inp[0] for inp in inputs]), np.array(labels).reshape(-1, 1)

        assert inputs_context.shape[0] == inputs_target.shape[0]
        assert inputs_context.shape[0] == labels.shape[0]

        # print(inputs_context.shape, inputs_target.shape, labels.shape)
        for eg_id_start in range(0, inputs_context.shape[0], batch_size):
            yield (
                inputs_target[eg_id_start: min(eg_id_start + batch_size, inputs_target.shape[0])],
                inputs_context[eg_id_start: min(eg_id_start + batch_size, inputs_context.shape[0]), :]
            ), labels[eg_id_start: min(eg_id_start + batch_size, labels.shape[0])]


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
K.clear_session()

cbow_validation_callback = ValidationCallback(valid_term_ids, cbow_model, tokenizer)

for ei in range(epochs):
    print(f"Epoch: {ei+1}/{epochs} started")
    news_cbow_gen = cbow_data_generator(news_sequences, window_size, batch_size, negative_samples)
    cbow_model.fit(
        news_cbow_gen,
        epochs=1,
        callbacks=cbow_validation_callback,
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

save_embeddings(cbow_model, tokenizer, n_vocab, save_dir='cbow_embeddings')


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


cbow_context_embeddings = pd.read_pickle(
    os.path.join('cbow_embeddings', 'context_embedding.pkl')
)

cbow_words, cbow_embeddings = np.array(cbow_context_embeddings.index), cbow_context_embeddings.values

tsne = TSNE(perplexity=30, n_components=2, n_iter=5000, metric='cosine')

print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold

cbow_embeddings_norm = cbow_embeddings / np.sum(cbow_embeddings**2, keepdims=True, axis=1)
cbow_tsne_embeddings = tsne.fit_transform(cbow_embeddings_norm)

plot(cbow_tsne_embeddings[1:1501], cbow_words[1:1501])

