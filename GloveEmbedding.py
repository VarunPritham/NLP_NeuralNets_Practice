from six.moves.urllib.request import urlretrieve
import zipfile
import numpy as np
import pandas as pd
import os
import time
import random
import tensorflow as tf
from matplotlib import pylab
from scipy.sparse import lil_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from scipy.sparse import save_npz, load_npz
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Embedding, Dot, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'


def download_data(url, data_dir):
    """Download a file if not present, and make sure it's the right size."""

    # Create the data directory if not exist
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, 'bbc-fulltext.zip')

    # If file doesnt exist, download
    if not os.path.exists(file_path):
        print('Downloading file...')
        filename, _ = urlretrieve(url, file_path)
    else:
        print("File already exists")

    extract_path = os.path.join(data_dir, 'bbc')

    # If data has not been extracted already, extract data
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

n_vocab = 15000 + 1
tokenizer = Tokenizer(
    num_words=n_vocab - 1,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ', oov_token=''
)

tokenizer.fit_on_texts(news_stories)
print("Data fitted on the tokenizer")


def generate_cooc_matrix(text, tokenizer, window_size, n_vocab, use_weighting=True):
    # Convert list of text to list of list of word IDs
    sequences = tokenizer.texts_to_sequences(text)

    # A sparse matrix to retain co-occurrences of words
    cooc_mat = lil_matrix((n_vocab, n_vocab), dtype=np.float32)

    # Go through each sequence one by one
    for si, sequence in enumerate(sequences):

        # Printing the progress
        if (si + 1) % 100 == 0:
            print('.' * ((si + 1) // 100), f"{si + 1}/{len(sequences)}", end='\r')

        # For each target word,
        for i, wi in zip(np.arange(window_size, len(sequence) - window_size), sequence[window_size:-window_size]):

            # Get the context window word IDs
            context_window = sequence[i - window_size: i + window_size + 1]

            # The weight for the words in the context window (except target word) will be 1
            window_weights = np.ones(shape=(window_size * 2 + 1,), dtype=np.float32)
            window_weights[window_size] = 0.0

            if use_weighting:
                # If weighting is used, penalize context words based on distance to target word
                distances = np.abs(np.arange(-window_size, window_size + 1))
                distances[window_size] = 1.0
                # Update the sparse matrix
                cooc_mat[wi, context_window] += window_weights / distances
            else:
                # Update the sparse matrix
                cooc_mat[wi, context_window] += window_weights

    print("\n")

    return cooc_mat


# ----------------------------------------- IMPORTANT ---------------------------------------------- #
#                                                                                                    #
# Set this true or false, depending on whether you want to generate the matrix or reuse the existing #
#                                                                                                    #
# ---------------------------------------------------------------------------------------------------#
generate_cooc = True

# Generate the matrix
if generate_cooc:
    t1 = time.time()
    cooc_mat = generate_cooc_matrix(news_stories, tokenizer, 1, n_vocab, True)
    t2 = time.time()
    print(f"It took {t2 - t1} seconds to generate the co-occurrence matrix")

    save_npz(os.path.join('data', 'cooc_mat.npz'), cooc_mat.tocsr())
# Load the matrix from disk
else:
    try:
        cooc_mat = load_npz(os.path.join('data', 'cooc_mat.npz')).tolil()
        print(f"Cooc matrix of type {type(cooc_mat).__name__} was loaded from disk")
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            "Could not find the co-occurrence matrix on the disk. Did you generate the matrix by setting generate_cooc=True?"
        )


word = 'sport'
assert word in tokenizer.word_index, f"Word {word} is not in the tokenizer"
assert tokenizer.word_index[word] <= n_vocab, f"The word {word} is an out of vocabuary word. Please try something else"

# Get the vector of co-occurrences for a given word
cooc_vec = np.array(cooc_mat.getrow(tokenizer.word_index[word]).todense()).ravel()
# Get indices of words with maximum value
max_ind = np.argsort(cooc_vec)[-25:]

# Plot the words and values
plt.figure(figsize=(16,8))
plt.bar(np.arange(0, 25), cooc_vec[max_ind])
plt.xticks(ticks=np.arange(0, 25), labels=[tokenizer.index_word[i] for i in max_ind], rotation=60)

batch_size = 4096 # Data points in a single batch

embedding_size = 128 # Dimension of the embedding vector.

window_size=1 # We use a window size of 1 on either side of target word

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

# Define two input layers for context and target words
word_i = Input(shape=())
word_j = Input(shape=())

# Each context and target has their own embeddings (weights and biases)

# Embedding weights
embeddings_i = Embedding(n_vocab, embedding_size, name='target_embedding')(word_i)
embeddings_j = Embedding(n_vocab, embedding_size, name='context_embedding')(word_j)

# Embedding biases
b_i = Embedding(n_vocab, 1, name='target_embedding_bias')(word_i)
b_j = Embedding(n_vocab, 1, name='context_embedding_bias')(word_j)

# Compute the dot product between embedding vectors (i.e. w_i.w_j)
ij_dot = Dot(axes=-1)([embeddings_i,embeddings_j])

# Add the biases (i.e. w_i.w_j + b_i + b_j )
pred = Add()([ij_dot, b_i, b_j])

# The final model
glove_model = Model(inputs=[word_i, word_j],outputs=pred, name='glove_model')

# Glove has a specific loss function with a sound mathematical underpinning
# It is a form of mean squared error
glove_model.compile(loss="mse", optimizer = 'adam')

glove_model.summary()

news_sequences = tokenizer.texts_to_sequences(news_stories)


def glove_data_generator(
        sequences, window_size, batch_size, vocab_size, cooccurrence_matrix, x_max=100.0, alpha=0.75, seed=None
):
    """ Generate batches of inputs and targets for GloVe """

    # Shuffle the data so that, every epoch, the order of data is different
    rand_sequence_ids = np.arange(len(sequences))
    np.random.shuffle(rand_sequence_ids)

    # We will use a sampling table to make sure, we don't oversample stopwords
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # For each story/article
    for si in rand_sequence_ids:

        # Generate positive skip-grams while using sub-sampling
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequences[si],
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0.0,
            shuffle=False,
            sampling_table=sampling_table,
            seed=seed
        )

        # Take targets and context words separately
        targets, context = zip(*positive_skip_grams)
        targets, context = np.array(targets).ravel(), np.array(context).ravel()

        x_ij = np.array(cooccurrence_matrix[targets, context].toarray()).ravel()

        # Compute log - Introducing an additive shift to make sure we don't compute log(0)
        log_x_ij = np.log(x_ij + 1)

        # Sample weights
        # if x < x_max => (x/x_max)**alpha / else => 1
        sample_weights = np.where(x_ij < x_max, (x_ij / x_max) ** alpha, 1)

        # If seed is not provided generate a random one
        if not seed:
            seed = random.randint(0, 10e6)

        # Shuffle data
        np.random.seed(seed)
        np.random.shuffle(context)
        np.random.seed(seed)
        np.random.shuffle(targets)
        np.random.seed(seed)
        np.random.shuffle(log_x_ij)
        np.random.seed(seed)
        np.random.shuffle(sample_weights)

        # Generate a batch or data in the format
        # ((target words, context words), log(X_ij) <- true targets, f(X_ij) <- sample weights)
        for eg_id_start in range(0, context.shape[0], batch_size):
            yield (
                targets[eg_id_start: min(eg_id_start + batch_size, targets.shape[0])],
                context[eg_id_start: min(eg_id_start + batch_size, context.shape[0])]
            ), log_x_ij[eg_id_start: min(eg_id_start + batch_size, x_ij.shape[0])], \
                sample_weights[eg_id_start: min(eg_id_start + batch_size, sample_weights.shape[0])]

news_glove_data_gen = glove_data_generator(
    news_sequences, 2, 10, n_vocab, cooc_mat
)


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
            similar_word_str = ', '.join([self.tokenizer.index_word[j] for j in similarity_top_k[i, :] if j > 1])
            print(f"{self.tokenizer.index_word[term_id]}: {similar_word_str}")

        print('\n')


glove_validation_callback = ValidationCallback(valid_term_ids, glove_model, tokenizer)

# Train the model for several epochs
for ei in range(epochs):
    print(f"Epoch: {ei + 1}/{epochs} started")

    news_glove_data_gen = glove_data_generator(
        news_sequences, window_size, batch_size, n_vocab, cooc_mat
    )

    glove_model.fit(
        news_glove_data_gen, epochs=1,
        callbacks=glove_validation_callback,
    )


def save_embeddings(model, tokenizer, vocab_size, save_dir):
    # Create the directory if doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get the words sorted according to their ID from the tokenizer
    _, words_sorted = zip(*sorted(list(tokenizer.index_word.items()), key=lambda x: x[0])[:vocab_size - 1])
    # Add one word in front to represent the reserved ID (0)
    words_sorted = [None] + list(words_sorted)

    # Create a new array by concatenating embeddings and bias

    context_embedding_weights = model.get_layer("context_embedding").get_weights()[0]
    context_embedding_bias = model.get_layer("context_embedding_bias").get_weights()[0]
    context_embedding = np.concatenate([context_embedding_weights, context_embedding_bias], axis=1)

    target_embedding_weights = model.get_layer("target_embedding").get_weights()[0]
    target_embedding_bias = model.get_layer("target_embedding_bias").get_weights()[0]
    target_embedding = np.concatenate([target_embedding_weights, target_embedding_bias], axis=1)

    # Save the array as a Pandas DataFrames
    pd.DataFrame(
        context_embedding,
        index=words_sorted
    ).to_pickle(os.path.join(save_dir, "context_embedding_and_bias.pkl"))

    pd.DataFrame(
        target_embedding,
        index=words_sorted
    ).to_pickle(os.path.join(save_dir, "target_embedding_and_bias.pkl"))


save_embeddings(glove_model, tokenizer, n_vocab, save_dir='glove_embeddings')