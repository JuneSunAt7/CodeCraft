import os
import parso
import re
from collections import Counter
import math
import numpy as np

def preprocess_code(code):

    code_no_comments = re.sub(r'#.*?\n', '\n', code)

    code_with_docstrings = re.sub(r'"""(.*?)"""', r'\1', code_no_comments, flags=re.DOTALL)

    tokens = re.findall(r'\b\w+\b|[^\w\s]', code_with_docstrings)
    tokens = [token for token in tokens if token.strip()]
    return ' '.join(tokens).strip()


def load_and_preprocess_data(directory):
    code_samples = []
    empty_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        code = f.read()
                        if not code.strip():
                            empty_files.append(file)
                            continue

                        processed_code = preprocess_code(code)
                        if processed_code:
                            code_samples.append(processed_code)
                        else:
                            print(f"File {file} resulted in empty code after preprocessing.")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    if empty_files:
        print(f"Ignored {len(empty_files)} empty files:")
        for file in empty_files[:5]:
            print(f"File {file} is empty.")

    return code_samples

directory = 'assets'
if not os.path.exists(directory) or not os.listdir(directory):
    raise ValueError(f"No .py files found in directory: {directory}")

code_samples = load_and_preprocess_data(directory)

if not code_samples:
    raise ValueError("No code samples found. Check the directory path.")

print(f"Loaded {len(code_samples)} code samples.")

# Токенизация
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(code_samples)

print(f"Total unique words before filtering: {len(tokenizer.word_index)}")

# filter
word_counts = Counter()

for line in code_samples:
    words = line.split()  # str to words
    word_counts.update(words)

min_frequency = 1

# create new tokenizer
filtered_code_samples = [' '.join([word for word in line.split() if word_counts[word] >= min_frequency]) for line in
                         code_samples]

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(filtered_code_samples)
import pickle

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

vocab_size = len(tokenizer.word_index) + 1  # + OOV token
print(f"Total unique words after filtering: {len(tokenizer.word_index)}")
print(f"Vocab size used in model: {vocab_size}")

input_sequences = []
short_lines = []

for i, line in enumerate(filtered_code_samples):
    token_list = tokenizer.texts_to_sequences([line])[0]

    if not token_list or len(token_list) < 2:
        short_lines.append((i, line))
        continue

    for j in range(1, len(token_list)):
        n_gram_sequence = token_list[:j + 1]
        # check indexes
        if all(index < vocab_size for index in n_gram_sequence):
            input_sequences.append(n_gram_sequence)

if len(input_sequences) == 0:
    print("Input sequences are empty. Debugging info:")
    print(f"Total code samples: {len(filtered_code_samples)}")
    print(f"Samples with less than 2 tokens: {len(short_lines)}")
    raise ValueError("Input sequences are empty.")

max_sequence_len = 50

# inputs & labels
xs = pad_sequences(input_sequences, maxlen=max_sequence_len - 1, padding='pre')
ys = np.array([seq[-1] for seq in input_sequences])

print(f"Max token index in xs: {np.max(xs)}")
print(f"Vocab size: {vocab_size}")

if np.max(xs) >= vocab_size:
    raise ValueError("Some indices in xs are out of bounds.")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(vocab_size, max_len, embed_dim=64, num_heads=1, ff_dim=256):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(vocab_size, embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer, training=True)
    flatten = tf.keras.layers.Flatten()(transformer_block)
    outputs = Dense(vocab_size, activation="softmax")(flatten)
    model = Model(inputs, outputs)
    return model

max_len = max_sequence_len - 1
embed_dim = 64
num_heads = 1
ff_dim = 256  # less hidden layer

model = build_transformer_model(vocab_size, max_len, embed_dim, num_heads, ff_dim)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

from tensorflow.keras.utils import Sequence


class CodeDataGenerator(Sequence):
    def __init__(self, xs, ys, batch_size=8):
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.xs) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.xs))
        batch_x = self.xs[start:end]
        batch_y = self.ys[start:end]
        return batch_x, batch_y

batch_size = 8
if len(xs) < batch_size:
    raise ValueError(f"Not enough data for batch size {batch_size}. Data size: {len(xs)}")

data_generator = CodeDataGenerator(xs, ys, batch_size=batch_size)

epochs = 4
model.fit(data_generator, epochs=epochs)

model.save('code_autocompletion.h5')
# func for prediction
def generate_code(seed_text, next_words=10, max_sequence_len=max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0).argmax(axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# final testing
seed_text = "def hello_world():"
generated_code = generate_code(seed_text, next_words=10)
print(generated_code)