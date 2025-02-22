import os
import parso
import re
from collections import Counter
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# filter for beginners
def filter_for_beginners(code):
    if "lambda" in code or "import" in code or "class" in code:
        return False
    return True

directory = 'Hello-World-main'
if not os.path.exists(directory) or not os.listdir(directory):
    raise ValueError(f"No .py files found in directory: {directory}")

code_samples = load_and_preprocess_data(directory)
filtered_code_samples = [code for code in code_samples if filter_for_beginners(code)]

if not filtered_code_samples:
    raise ValueError("No filtered code samples found.")

print(f"Loaded {len(filtered_code_samples)} filtered code samples.")

# label of lang
filtered_code_samples = ['[PYTHON] ' + code for code in filtered_code_samples]

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(filtered_code_samples)

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocab size used in model: {vocab_size}")

input_sequences = []
for line in filtered_code_samples:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = 100
xs = pad_sequences(input_sequences, maxlen=max_sequence_len - 1, padding='pre')
ys = np.array([seq[-1] for seq in input_sequences])

embed_dim = 256
num_heads = 8
ff_dim = 1024
dropout_rate = 0.3

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=dropout_rate):
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

def build_transformer_model(vocab_size, max_len, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(vocab_size, embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer, training=True)
    flatten = tf.keras.layers.Flatten()(transformer_block)
    outputs = Dense(vocab_size, activation="softmax")(flatten)
    model = Model(inputs, outputs)
    return model

model = build_transformer_model(vocab_size, max_len=max_sequence_len - 1)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

batch_size = 16
epochs = 20

from tensorflow.keras.utils import Sequence

class CodeDataGenerator(Sequence):
    def __init__(self, xs, ys, batch_size=batch_size):
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

data_generator = CodeDataGenerator(xs, ys, batch_size=batch_size)
model.fit(data_generator, epochs=epochs)

def generate_code(seed_text, next_words=10, max_sequence_len=max_sequence_len, top_k=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]
        top_k_indices = tf.random.categorical(tf.math.log([predictions]), num_samples=1)[0].numpy()[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == top_k_indices:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

seed_text = "# Hello world in Python !"
generated_code = generate_code(seed_text, next_words=10)
print(generated_code)