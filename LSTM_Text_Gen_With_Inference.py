import os
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'
dir_name = 'Data_gen'


def download_data(url, filename, download_dir):
    """Download a file if not present, and make sure it's the right size."""

    # Create directories if doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # If file doesn't exist download
    if not os.path.exists(os.path.join(download_dir, filename)):
        filepath, _ = urlretrieve(url + filename, os.path.join(download_dir, filename))
    else:
        filepath = os.path.join(download_dir, filename)

    return filepath


# Number of files and their names to download
num_files = 209
filenames = [format(i, '03d') + '.txt' for i in range(1, num_files + 1)]

# Download each file
for fn in filenames:
    download_data(url, fn, dir_name)

# Check if all files are downloaded
for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
    assert file_exists
print(f"{len(filenames)} files found.")

# Fix the random seed so we get the same outptu everytime
random_state = 54321

filenames = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]

# First separate train and valid+test data
train_filenames, test_and_valid_filenames = train_test_split(filenames, test_size=0.2, random_state=random_state)

# Separate valid+test data to validation and test data
valid_filenames, test_filenames = train_test_split(test_and_valid_filenames, test_size=0.5, random_state=random_state)

# Print out the sizes and some sample filenames
for subset_id, subset in zip(('train', 'valid', 'test'), (train_filenames, valid_filenames, test_filenames)):
    print(f"Got {len(subset)} files in the {subset_id} dataset (e.g. {subset[:3]})")

bigram_set = set()

# Go through each file in the training set
for fname in train_filenames:
    document = []  # This will hold all the text
    with open(fname, 'r', errors="ignore") as f:
        for row in f:
            # Convert text to lower case to reduce input dimensionality
            document.append(row.lower())

        # From the list of text we have, generate one long string (containing all training stories)
        document = " ".join(document)

        # Update the set with all bigrams found
        bigram_set.update([document[i:i + 2] for i in range(0, len(document), 2)])

# Assign to a variable and print
n_vocab = len(bigram_set)
print(f"Found {n_vocab} unique bigrams")


def generate_tf_dataset(filenames, ngram_width, window_size, batch_size, shuffle=False):
    """ Generate batched data from a list of files speficied """

    # Read the data found in the documents
    documents = []
    for f in filenames:
        doc = tf.io.read_file(f)
        doc = tf.strings.ngrams(  # Generate ngrams from the string
            tf.strings.bytes_split(  # Create a list of chars from a string
                tf.strings.regex_replace(  # Replace new lines with space
                    tf.strings.lower(  # Convert string to lower case
                        doc
                    ), "\n", " "
                )
            ),
            ngram_width, separator=''
        )
        documents.append(doc.numpy().tolist())

    # documents is a list of list of strings, where each string is a story
    # From that we generate a ragged tensor
    documents = tf.ragged.constant(documents)
    # Create a dataset where each row in the ragged tensor would be a sample
    doc_dataset = tf.data.Dataset.from_tensor_slices(documents)
    # We need to perform a quick transformation - tf.strings.ngrams would generate
    # all the ngrams (e.g. abcd -> ab, bc, cd) with overlap, however for our data
    # we do not need the overlap, so we need to skip the overlapping ngrams
    # the following line does that
    doc_dataset = doc_dataset.map(lambda x: x[::ngram_width])

    # Here we are using a window function to generate windows from text
    # For a text sequence with window_size 3 and shift 1 you get
    # e.g. ab, cd, ef, gh, ij, ... -> [ab, cd, ef], [cd, ef, gh], [ef, gh, ij], ...
    # each of these windows is a single training sequence for our model
    doc_dataset = doc_dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(
            x
        ).window(
            size=window_size + 1, shift=int(window_size * 0.75)
        ).flat_map(
            lambda window: window.batch(window_size + 1, drop_remainder=True)
        )
    )

    # From each windowed sequence we generate input and target tuple
    # e.g. [ab, cd, ef] -> ([ab, cd], [cd, ef])
    doc_dataset = doc_dataset.map(lambda x: (x[:-1], x[1:]))

    # Shuffle the data if required
    doc_dataset = doc_dataset.shuffle(buffer_size=batch_size * 10) if shuffle else doc_dataset

    # Batch the data
    doc_dataset = doc_dataset.batch(batch_size=batch_size)

    # Return the data
    return doc_dataset

ngram_length = 2
batch_size = 128
window_size = 128

train_ds = generate_tf_dataset(train_filenames, ngram_length, window_size, batch_size, shuffle=True)
valid_ds = generate_tf_dataset(valid_filenames, ngram_length, window_size, batch_size)
test_ds = generate_tf_dataset(test_filenames, ngram_length, window_size, batch_size)

K.clear_session()
tf.compat.v1.reset_default_graph()

text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=n_vocab, standardize=None,
    split=None, input_shape=(window_size,)
)

# Train the model on existing data
text_vectorizer.adapt(train_ds)

train_ds = train_ds.map(lambda x, y: (x, text_vectorizer(y)))
valid_ds = valid_ds.map(lambda x, y: (x, text_vectorizer(y)))
test_ds = test_ds.map(lambda x, y: (x, text_vectorizer(y)))

lm_model = models.Sequential([
    text_vectorizer,
    layers.Embedding(n_vocab+2, 96),
    layers.LSTM(512, return_sequences=True),
    layers.LSTM(256, return_sequences=True),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_vocab, activation='softmax')
])


class PerplexityMetric(tf.keras.metrics.Mean):

    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def _calculate_perplexity(self, real, pred):
        # The next 4 lines zero-out the padding from loss calculations,
        # this follows the logic from: https://www.tensorflow.org/beta/tutorials/text/transformer#loss_and_metrics
        loss_ = self.cross_entropy(real, pred)

        # Calculating the perplexity steps:
        step1 = K.mean(loss_, axis=-1)
        perplexity = K.exp(step1)

        return perplexity

    def update_state(self, y_true, y_pred, sample_weight=None):
        perplexity = self._calculate_perplexity(y_true, y_pred)
        # Remember self.perplexity is a tensor (tf.Variable), so using simply "self.perplexity = perplexity" will result in error because of mixing EagerTensor and Graph operations
        super().update_state(perplexity)

lm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', PerplexityMetric()])
lm_model.summary()

lstm_history = lm_model.fit(train_ds, validation_data=valid_ds, epochs=50)

lm_model.evaluate(test_ds)
# Define inputs to the model
inp = tf.keras.layers.Input(dtype=tf.string, shape=(1,))

text_vectorized_out = lm_model.get_layer('text_vectorization')(inp)

inp_state_c_lstm = tf.keras.layers.Input(shape=(512,))
inp_state_h_lstm = tf.keras.layers.Input(shape=(512,))
inp_state_c_lstm_1 = tf.keras.layers.Input(shape=(256,))
inp_state_h_lstm_1 = tf.keras.layers.Input(shape=(256,))

# Define embedding layer and output
emb_layer = lm_model.get_layer('embedding')
emb_out = emb_layer(text_vectorized_out)

# Defining a LSTM layers and output
lstm_layer = tf.keras.layers.LSTM(512, return_state=True, return_sequences=True)
lstm_out, lstm_state_c, lstm_state_h = lstm_layer(emb_out, initial_state=[inp_state_c_lstm, inp_state_h_lstm])

lstm_1_layer = tf.keras.layers.LSTM(256, return_state=True, return_sequences=True)
lstm_1_out, lstm_1_state_c, lstm_1_state_h = lstm_1_layer(lstm_out, initial_state=[inp_state_c_lstm_1, inp_state_h_lstm_1])

# Defining a Dense layer and output
dense_out = lm_model.get_layer('dense')(lstm_1_out)

# Defining the final Dense layer and output
final_out = lm_model.get_layer('dense_1')(dense_out)
#softmax_out = tf.keras.layers.Activation(activation='softmax')(final_out)

# Copy the weights from the original model
lstm_layer.set_weights(lm_model.get_layer('lstm').get_weights())
lstm_1_layer.set_weights(lm_model.get_layer('lstm_1').get_weights())

# Define final model
infer_model = tf.keras.models.Model(
    inputs=[inp, inp_state_c_lstm, inp_state_h_lstm, inp_state_c_lstm_1, inp_state_h_lstm_1],
    outputs=[final_out, lstm_state_c, lstm_state_h, lstm_1_state_c, lstm_1_state_h])


# Summary
infer_model.summary()

text = ["When adam and eve were driven out of paradise, they were compelled to build a house for themselves on barren ground"]

seq = [text[0][i:i+2] for i in range(0, len(text[0]), 2)]

# build up model state using the given string
print(f"Making predictions from a {len(seq)} element long input")

vocabulary = infer_model.get_layer("text_vectorization").get_vocabulary()
index_word = dict(zip(range(len(vocabulary)), vocabulary))


infer_model.reset_states()
# Definin the initial state as all zeros
state_c = np.zeros(shape=(1, 512))
state_h = np.zeros(shape=(1, 512))
state_c_1 = np.zeros(shape=(1, 256))
state_h_1 = np.zeros(shape=(1, 256))

# Recursively update the model by assining new state to state
for c in seq:
    # print(c)
    out, state_c, state_h, state_c_1, state_h_1 = infer_model.predict(
        [np.array([[c]]), state_c, state_h, state_c_1, state_h_1]
    )

# Get final prediction after feeding the input string
wid = int(np.argmax(out[0], axis=-1).ravel())
word = index_word[wid]
text.append(word)

# Define first input to generate text recursively from
x = np.array([[word]])

# Code listing 10.7
for _ in range(500):

    # Get the next output and state
    out, state_c, state_h, state_c_1, state_h_1 = infer_model.predict([x, state_c, state_h, state_c_1, state_h_1])

    # Get the word id and the word from out
    out_argsort = np.argsort(out[0], axis=-1).ravel()
    wid = int(out_argsort[-1])
    word = index_word[wid]

    # If the word ends with space, we introduce a bit of randomness
    # Essentially pick one of the top 3 outputs for that timestep depending on their likelihood
    if word.endswith(' '):
        if np.random.normal() > 0.5:
            width = 5
            i = np.random.choice(list(range(-width, 0)), p=out_argsort[-width:] / out_argsort[-width:].sum())
            wid = int(out_argsort[i])
            word = index_word[wid]

    # Append the prediction
    text.append(word)

    # Recursively make the current prediction the next input
    x = np.array([[word]])

# Print the final output
print('\n')
print('=' * 60)
print("Final text: ")
print(''.join(text))