import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from bleu import compute_bleu
import matplotlib.pyplot as plt


def fix_random_seed(seed):
    """ Setting the random seed of various libraries """
    try:
        np.random.seed(seed)
    except NameError:
        print("Warning: Numpy is not imported. Setting the seed for Numpy failed.")
    try:
        tf.random.set_seed(seed)
    except NameError:
        print("Warning: TensorFlow is not imported. Setting the seed for TensorFlow failed.")
    try:
        random.seed(seed)
    except NameError:
        print("Warning: random module is not imported. Setting the seed for random failed.")


# Fixing the random seed
random_seed = 4321
fix_random_seed(random_seed)

print(f"TensorFlow version: {tf.__version__}")

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU found!")
    pass

n_sentences = 200000

# Loading English sentences
original_en_sentences = []
with open(os.path.join('data_nmt', 'train.en'), 'r', encoding='utf-8') as en_file:
    for i, row in enumerate(en_file):
        if i < 50: continue  # or i==22183 or i==27781 or i==81827: continue
        if i >= n_sentences: break
        original_en_sentences.append(row.strip().split(" "))

# Loading German sentences
original_de_sentences = []
with open(os.path.join('data_nmt', 'train.de'), 'r', encoding='utf-8') as de_file:
    for i, row in enumerate(de_file):
        if i < 50: continue  # or i==22183 or i==27781 or i==81827: continue
        if i >= n_sentences: break
        original_de_sentences.append(row.strip().split(" "))


en_sentences = [["<s>"]+sent+["</s>"] for sent in original_en_sentences]
de_sentences = [["<s>"]+sent+["</s>"] for sent in original_de_sentences]

train_en_sentences, valid_test_en_sentences, train_de_sentences, valid_test_de_sentences = train_test_split(
    np.array(en_sentences), np.array(de_sentences), test_size=0.2
)

valid_en_sentences, test_en_sentences, valid_de_sentences, test_de_sentences = train_test_split(
    valid_test_en_sentences, valid_test_de_sentences, test_size=0.5)

print(f"Train size: {len(train_en_sentences)}")
print(f"Valid size: {len(valid_en_sentences)}")
print(f"Test size: {len(test_en_sentences)}")

print("Sequence lengths (English)")
print(pd.Series(train_en_sentences).str.len().describe(percentiles=[0.2, 0.5, 0.8]))

print("Sequence lengths (German)")
print(pd.Series(train_de_sentences).str.len().describe(percentiles=[0.2, 0.5, 0.8]))

n_en_seq_length = 36
n_de_seq_length = 33

pad_token = '<pad>'

train_en_sentences_padded = pad_sequences(train_en_sentences, maxlen=n_en_seq_length, value=pad_token, dtype=object, truncating='post', padding='pre')
valid_en_sentences_padded = pad_sequences(valid_en_sentences, maxlen=n_en_seq_length, value=pad_token, dtype=object, truncating='post', padding='pre')
test_en_sentences_padded = pad_sequences(test_en_sentences, maxlen=n_en_seq_length, value=pad_token, dtype=object, truncating='post', padding='pre')

train_de_sentences_padded = pad_sequences(train_de_sentences, maxlen=n_de_seq_length, value=pad_token, dtype=object, truncating='post', padding='post')
valid_de_sentences_padded = pad_sequences(valid_de_sentences, maxlen=n_de_seq_length, value=pad_token, dtype=object, truncating='post', padding='post')
test_de_sentences_padded = pad_sequences(test_de_sentences, maxlen=n_de_seq_length, value=pad_token, dtype=object, truncating='post', padding='post')

n_vocab = 25000 + 1

en_vocabulary = []
with open(os.path.join('data', 'vocab.50K.en'), 'r', encoding='utf-8') as en_file:
    for ri, row in enumerate(en_file):
        if ri >= n_vocab: break

        en_vocabulary.append(row.strip())

de_vocabulary = []
with open(os.path.join('data', 'vocab.50K.de'), 'r', encoding='utf-8') as de_file:
    for ri, row in enumerate(de_file):
        if ri >= n_vocab: break

        de_vocabulary.append(row.strip())

en_unk_token = en_vocabulary.pop(0)
de_unk_token = de_vocabulary.pop(0)

en_lookup_layer = tf.keras.layers.StringLookup(
    oov_token=en_unk_token, vocabulary=en_vocabulary, mask_token=pad_token, pad_to_max_tokens=False
)

de_lookup_layer = tf.keras.layers.StringLookup(
    oov_token=de_unk_token, vocabulary=de_vocabulary, mask_token=pad_token, pad_to_max_tokens=False
)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # Weights to compute Bahdanau attention
        self.Wa = tf.keras.layers.Dense(units, use_bias=False)
        self.Ua = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention(use_scale=True)

    def call(self, query, key, value, mask, return_attention_scores=False):

        # Compute `Wa.ht`.
        wa_query = self.Wa(query)

        # Compute `Ua.hs`.
        ua_key = self.Ua(key)

        # Compute masks
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        # Compute the attention
        context_vector, attention_weights = self.attention(
            inputs=[wa_query, value, ua_key],
            mask=[query_mask, value_mask, value_mask],
            return_attention_scores=True,
        )

        if not return_attention_scores:
            return context_vector
        else:
            return context_vector, attention_weights


K.clear_session()

# Defining the encoder layers
encoder_input = tf.keras.layers.Input(shape=(n_en_seq_length,), dtype=tf.string)
# Converting tokens to IDs
encoder_wid_out = en_lookup_layer(encoder_input)

# Embedding layer and lookup
encoder_emb_out = tf.keras.layers.Embedding(len(en_lookup_layer.get_vocabulary()), 128, mask_zero=True)(encoder_wid_out)

# Encoder GRU layer
encoder_gru_out, encoder_gru_last_state = tf.keras.layers.GRU(256, return_sequences=True, return_state=True)(encoder_emb_out)

# Defining the encoder model: in - encoder_input / out - output of the GRU layer
encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoder_gru_out)

# Defining the decoder layers
decoder_input = tf.keras.layers.Input(shape=(n_de_seq_length-1,), dtype=tf.string)
# Converting tokens to IDs (Decoder)
decoder_wid_out = de_lookup_layer(decoder_input)

# Embedding layer and lookup (decoder)
full_de_vocab_size = len(de_lookup_layer.get_vocabulary())
decoder_emb_out = tf.keras.layers.Embedding(full_de_vocab_size, 128, mask_zero=True)(decoder_wid_out)
decoder_gru_out = tf.keras.layers.GRU(256, return_sequences=True)(decoder_emb_out, initial_state=encoder_gru_last_state)

# The attention mechanism (inputs: [q, v, k])
decoder_attn_out, attn_weights = BahdanauAttention(256)(
    query=decoder_gru_out, key=encoder_gru_out, value=encoder_gru_out,
    mask=(encoder_wid_out != 0),
    return_attention_scores=True
)

# Concatenate GRU output and the attention output
context_and_rnn_output = tf.keras.layers.Concatenate(axis=-1)([decoder_attn_out, decoder_gru_out])

# Final prediction layer (size of the vocabulary)
decoder_out = tf.keras.layers.Dense(full_de_vocab_size, activation='softmax')(context_and_rnn_output)

# Final seq2seq model
seq2seq_model = tf.keras.models.Model(inputs=[encoder.inputs, decoder_input], outputs=decoder_out)

# We will use this model later to visualize attention patterns
attention_visualizer = tf.keras.models.Model(inputs=[encoder.inputs, decoder_input], outputs=[attn_weights, decoder_out])

# Compiling the model with a loss and an optimizer
seq2seq_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')

# Print model summary
seq2seq_model.summary()


class BLEUMetric(object):

    def __init__(self, vocabulary, name='perplexity', **kwargs):
        """ Computes the BLEU score (Metric for machine translation) """
        super().__init__()
        self.vocab = vocabulary
        self.id_to_token_layer = StringLookup(vocabulary=self.vocab, invert=True)

    def calculate_bleu_from_predictions(self, real, pred):
        """ Calculate the BLEU score for targets and predictions """

        # Get the predicted token IDs
        pred_argmax = tf.argmax(pred, axis=-1)

        # Convert token IDs to words using the vocabulary and the StringLookup
        pred_tokens = self.id_to_token_layer(pred_argmax)
        real_tokens = self.id_to_token_layer(real)

        def clean_text(tokens):
            """ Clean padding and <s>/</s> tokens to only keep meaningful words """

            # 3. Strip the string of any extra white spaces
            translations_in_bytes = tf.strings.strip(
                # 2. Replace everything after the eos token with blank
                tf.strings.regex_replace(
                    # 1. Join all the tokens to one string in each sequence
                    tf.strings.join(
                        tf.transpose(tokens), separator=' '
                    ),
                    "<\/s>.*", ""),
            )

            # Decode the byte stream to a string
            translations = np.char.decode(
                translations_in_bytes.numpy().astype(np.bytes_), encoding='utf-8'
            )

            # If the string is empty, add a [UNK] token
            # Otherwise get a Division by zero error
            translations = [sent if len(sent) > 0 else en_unk_token for sent in translations]

            # Split the sequences to individual tokens
            translations = np.char.split(translations).tolist()

            return translations

        # Get the clean versions of the predictions and real seuqences
        pred_tokens = clean_text(pred_tokens)
        # We have to wrap each real sequence in a list to make use of a function to compute bleu
        real_tokens = [[token_seq] for token_seq in clean_text(real_tokens)]

        # The compute_bleu method accpets the translations and references in the following format
        # tranlation - list of list of tokens
        # references - list of list of list of tokens
        bleu, precisions, bp, ratio, translation_length, reference_length = compute_bleu(real_tokens, pred_tokens,
                                                                                         smooth=False)

        return bleu


def prepare_data(de_lookup_layer, train_xy, valid_xy, test_xy):
    """ Create a data dictionary from the dataframes containing data """

    data_dict = {}
    for label, data_xy in zip(['train', 'valid', 'test'], [train_xy, valid_xy, test_xy]):
        data_x, data_y = data_xy
        en_inputs = data_x
        de_inputs = data_y[:, :-1]
        de_labels = de_lookup_layer(data_y[:, 1:]).numpy()
        data_dict[label] = {'encoder_inputs': en_inputs, 'decoder_inputs': de_inputs, 'decoder_labels': de_labels}

    return data_dict


def shuffle_data(en_inputs, de_inputs, de_labels, shuffle_inds=None):
    """ Shuffle the data randomly (but all of inputs and labels at ones)"""

    if shuffle_inds is None:
        # If shuffle_inds are not passed create a shuffling automatically
        shuffle_inds = np.random.permutation(np.arange(en_inputs.shape[0]))
    else:
        # Shuffle the provided shuffle_inds
        shuffle_inds = np.random.permutation(shuffle_inds)

    # Return shuffled data
    return (en_inputs[shuffle_inds], de_inputs[shuffle_inds], de_labels[shuffle_inds]), shuffle_inds


def check_for_nans(loss, model, en_lookup_layer, de_lookup_layer,x,y):
    if np.isnan(loss):
        for r_i in range(len(y)):
            loss_sample, _ = model.evaluate([x[0][r_i:r_i + 1], x[1][r_i:r_i + 1]], y[r_i:r_i + 1], verbose=0)
            if np.isnan(loss_sample):

                print('=' * 25, 'nan detected', '=' * 25)
                print('train_batch', i, 'r_i', r_i)
                print('en_input ->', x[0][r_i].tolist())
                print('en_input_wid ->', en_lookup_layer(x[0][r_i]).numpy().tolist())
                print('de_input ->', x[1][r_i].tolist())
                print('de_input_wid ->', de_lookup_layer(x[1][r_i]).numpy().tolist())
                print('de_output_wid ->', y[r_i].tolist())

                if r_i > 0:
                    print('=' * 25, 'no-nan', '=' * 25)
                    print('en_input ->', x[0][r_i - 1].tolist())
                    print('en_input_wid ->', en_lookup_layer(x[0][r_i - 1]).numpy().tolist())
                    print('de_input ->', x[1][r_i - 1].tolist())
                    print('de_input_wid ->', de_lookup_layer(x[1][r_i - 1]).numpy().tolist())
                    print('de_output_wid ->', y[r_i - 1].tolist())
                    return
                else:
                    continue


def train_model(model, en_lookup_layer, de_lookup_layer, train_xy, valid_xy, test_xy, epochs, batch_size, shuffle=True,
                predict_bleu_at_training=False):
    """ Training the model and evaluating on validation/test sets """

    # Define the metric
    bleu_metric = BLEUMetric(de_vocabulary)
    bleu_log = []

    # Define the data
    data_dict = prepare_data(de_lookup_layer, train_xy, valid_xy, test_xy)

    shuffle_inds = None

    for epoch in range(epochs):

        # Reset metric logs every epoch
        if predict_bleu_at_training:
            bleu_log = []
        accuracy_log = []
        loss_log = []

        # =================================================================== #
        #                         Train Phase                                 #
        # =================================================================== #

        # Shuffle data at the beginning of every epoch
        if shuffle:
            (en_inputs_raw, de_inputs_raw, de_labels), shuffle_inds = shuffle_data(
                data_dict['train']['encoder_inputs'],
                data_dict['train']['decoder_inputs'],
                data_dict['train']['decoder_labels'],
                shuffle_inds
            )
        else:
            (en_inputs_raw, de_inputs_raw, de_labels) = (
                data_dict['train']['encoder_inputs'],
                data_dict['train']['decoder_inputs'],
                data_dict['train']['decoder_labels'],
            )
        # Get the number of training batches
        n_train_batches = en_inputs_raw.shape[0] // batch_size

        prev_loss = None
        # Train one batch at a time
        for i in range(n_train_batches):
            # Status update
            print(f"Training batch {i + 1}/{n_train_batches}", end='\r')

            # Get a batch of inputs (english and german sequences)
            x = [en_inputs_raw[i * batch_size:(i + 1) * batch_size], de_inputs_raw[i * batch_size:(i + 1) * batch_size]]
            # Get a batch of targets (german sequences offset by 1)
            y = de_labels[i * batch_size:(i + 1) * batch_size]

            loss, accuracy = model.evaluate(x, y, verbose=0)

            # Check if any samples are causing NaNs
            check_for_nans(loss, model, en_lookup_layer, de_lookup_layer,x,y)

            # Train for a single step
            model.train_on_batch(x, y)
            # Evaluate the model to get the metrics
            # loss, accuracy = model.evaluate(x, y, verbose=0)

            # Update the epoch's log records of the metrics
            loss_log.append(loss)
            accuracy_log.append(accuracy)

            if predict_bleu_at_training:
                # Get the final prediction to compute BLEU
                pred_y = model.predict(x)
                bleu_log.append(bleu_metric.calculate_bleu_from_predictions(y, pred_y))

        print("")
        print(f"\nEpoch {epoch + 1}/{epochs}")
        if predict_bleu_at_training:
            print(
                f"\t(train) loss: {np.mean(loss_log)} - accuracy: {np.mean(accuracy_log)} - bleu: {np.mean(bleu_log)}")
        else:
            print(f"\t(train) loss: {np.mean(loss_log)} - accuracy: {np.mean(accuracy_log)}")
        # =================================================================== #
        #                      Validation Phase                               #
        # =================================================================== #

        val_en_inputs = data_dict['valid']['encoder_inputs']
        val_de_inputs = data_dict['valid']['decoder_inputs']
        val_de_labels = data_dict['valid']['decoder_labels']

        val_loss, val_accuracy, val_bleu = evaluate_model(
            model, de_lookup_layer, val_en_inputs, val_de_inputs, val_de_labels, batch_size
        )

        # Print the evaluation metrics of each epoch
        print(f"\t(valid) loss: {val_loss} - accuracy: {val_accuracy} - bleu: {val_bleu}")

    # =================================================================== #
    #                      Test Phase                                     #
    # =================================================================== #

    test_en_inputs = data_dict['test']['encoder_inputs']
    test_de_inputs = data_dict['test']['decoder_inputs']
    test_de_labels = data_dict['test']['decoder_labels']

    test_loss, test_accuracy, test_bleu = evaluate_model(
        model, de_lookup_layer, test_en_inputs, test_de_inputs, test_de_labels, batch_size
    )

    print(f"\n(test) loss: {test_loss} - accuracy: {test_accuracy} - bleu: {test_bleu}")


def evaluate_model(model, de_lookup_layer, en_inputs_raw, de_inputs_raw, de_labels, batch_size):
    """ Evaluate the model on various metrics such as loss, accuracy and BLEU """

    # Define the metric
    bleu_metric = BLEUMetric(de_vocabulary)

    loss_log, accuracy_log, bleu_log = [], [], []
    # Get the number of batches
    n_batches = en_inputs_raw.shape[0] // batch_size
    print(" ", end='\r')

    # Evaluate one batch at a time
    for i in range(n_batches):
        # Status update
        print(f"Evaluating batch {i + 1}/{n_batches}", end='\r')

        # Get the inputs and targers
        x = [en_inputs_raw[i * batch_size:(i + 1) * batch_size], de_inputs_raw[i * batch_size:(i + 1) * batch_size]]
        y = de_labels[i * batch_size:(i + 1) * batch_size]

        # Get the evaluation metrics
        loss, accuracy = model.evaluate(x, y, verbose=0)
        # Get the predictions to compute BLEU
        pred_y = model.predict(x)

        # Update logs
        loss_log.append(loss)
        accuracy_log.append(accuracy)
        bleu_log.append(bleu_metric.calculate_bleu_from_predictions(y, pred_y))

    return np.mean(loss_log), np.mean(accuracy_log), np.mean(bleu_log)

epochs = 10
batch_size = 72

t1 = time.time()
train_model(
    seq2seq_model,
    en_lookup_layer, de_lookup_layer,
    (train_en_sentences_padded, train_de_sentences_padded),
    (valid_en_sentences_padded, valid_de_sentences_padded),
    (test_en_sentences_padded, test_de_sentences_padded),
    epochs,
    batch_size,
    shuffle=False
)
t2 = time.time()

print(f"\nIt took {t2-t1} seconds to complete the training")


def get_attention_matrix_for_sampled_data(attention_model, target_lookup_layer, test_xy, n_samples=5):
    test_x, test_y = test_xy

    rand_ids = np.random.randint(0, len(test_xy[0]), size=(n_samples,))
    print(rand_ids)
    results = []

    for rid in rand_ids:
        en_input = test_x[rid:rid + 1]
        de_input = test_y[rid:rid + 1, :-1]

        clean_en_input = []
        en_start_i = 0
        for i, w in enumerate(en_input.ravel()):
            if w == '<pad>':
                en_start_i = i + 1
                continue

            clean_en_input.append(w)
            if w == '</s>': break

        attn_weights, predictions = attention_model.predict([en_input, de_input])
        predicted_word_ids = np.argmax(predictions, axis=-1).ravel()
        predicted_words = [target_lookup_layer.get_vocabulary()[wid] for wid in predicted_word_ids]

        clean_predicted_words = []
        for w in predicted_words:
            clean_predicted_words.append(w)
            if w == '</s>': break

        results.append(
            {
                "attention_weights": attn_weights[
                                     0, :len(clean_predicted_words), en_start_i:en_start_i + len(clean_en_input)
                                     ],
                "input_words": clean_en_input,
                "predicted_words": clean_predicted_words
            }
        )

    return results


_, axes = plt.subplots(5, 1, figsize=(24, 40))

attention_results = get_attention_matrix_for_sampled_data(
    attention_visualizer,
    de_lookup_layer,
    (test_en_sentences_padded, test_de_sentences_padded),
    n_samples=5
)

for ax, result in zip(axes, attention_results):
    ax.imshow(result["attention_weights"])
    x_labels = result["input_words"]
    y_labels = result["predicted_words"]
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, rotation=0)


# Defining the encoder layers
encoder_input = tf.keras.layers.Input(shape=(n_en_seq_length,), dtype=tf.string)
# Converting tokens to IDs
en_lookup_layer = seq2seq_model.get_layer("string_lookup")
encoder_wid_out = en_lookup_layer(encoder_input)

# Embedding layer and lookup
en_emb_layer = seq2seq_model.get_layer("embedding")
encoder_emb_out = en_emb_layer(encoder_wid_out)

# Encoder GRU layer
en_gru_layer = seq2seq_model.get_layer("gru")
encoder_gru_out, encoder_gru_last_state = en_gru_layer(encoder_emb_out)

# Defining the encoder model: in - encoder_input / out - output of the GRU layer
encoder_model = tf.keras.models.Model(inputs=encoder_input, outputs=[encoder_gru_out, encoder_gru_last_state])

# Defining the decoder layers
decoder_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
decoder_init_state_input = tf.keras.layers.Input(shape=(256,))
encoder_state_input = tf.keras.layers.Input(shape=(n_en_seq_length,256))
encoder_input_mask = tf.keras.layers.Input(shape=(n_en_seq_length,), dtype=tf.bool)

# Converting tokens to IDs (Decoder)
de_lookup_layer = seq2seq_model.get_layer("string_lookup_1")
decoder_wid_out = de_lookup_layer(decoder_input)

# Embedding layer and lookup (decoder)
de_emb_layer = seq2seq_model.get_layer("embedding_1")
decoder_emb_out = de_emb_layer(decoder_wid_out)

de_gru_layer = tf.keras.layers.GRU(256, return_sequences=True)
decoder_gru_out = de_gru_layer(decoder_emb_out, initial_state=decoder_init_state_input)

# The attention mechanism (inputs: [q, v, k])
attention_layer = seq2seq_model.get_layer("bahdanau_attention")
decoder_attn_out, attn_weights = attention_layer(
    query=decoder_gru_out, key=encoder_state_input, value=encoder_state_input,
    mask=encoder_input_mask,
    return_attention_scores=True
)

# Concatenate GRU output and the attention output
context_and_rnn_output = tf.keras.layers.Concatenate(axis=-1)([decoder_attn_out, decoder_gru_out])

# Final prediction layer (size of the vocabulary)
de_dense_layer = seq2seq_model.get_layer("dense_2")
decoder_out = de_dense_layer(context_and_rnn_output)

# Final seq2seq model
decoder_model = tf.keras.models.Model(
    inputs=[decoder_input, decoder_init_state_input, encoder_state_input, encoder_input_mask],
    outputs=[decoder_out, decoder_gru_out]
)

decoder_model.compile()
de_gru_layer.set_weights(seq2seq_model.get_layer("gru_1").get_weights())


def generate_translation(en_sentence, en_lookup_layer, encoder_model, de_lookup_layer, decoder_model):
    de_vocabulary = de_lookup_layer.get_vocabulary()
    en_out, de_gru_state = encoder_model(en_sentence)

    y_pred = np.array([["<s>"]])
    predicted_sentence = [y_pred[0][0]]

    for _ in range(100):

        if y_pred == "</s>":
            break

        y_pred_probs, de_gru_state = decoder_model.predict([y_pred, de_gru_state, en_out, (en_sentence != pad_token)])
        de_gru_state = de_gru_state[:, 0, :]
        y_pred_wid = np.argmax(y_pred_probs, axis=-1).ravel()[0]
        y_pred = np.array([[de_vocabulary[y_pred_wid]]])
        predicted_sentence.append(y_pred[0][0])

    return ' '.join(predicted_sentence)


for en_sentence, de_sentence in zip(test_en_sentences_padded[:5, :], test_de_sentences_padded[:5, :]):
    en_sentence_string = ' '.join([en_word for en_word in en_sentence if en_word != pad_token])
    print(f"EN: {en_sentence_string}")
    de_sentence_string = ' '.join([de_word for de_word in de_sentence if de_word != pad_token])
    print(f"DE (true): {de_sentence_string}")

    de_predicted = generate_translation(
        en_sentence.reshape(1, -1), en_lookup_layer, encoder_model, de_lookup_layer, decoder_model
    )
    print(f"DE (predicted): {de_predicted}\n")