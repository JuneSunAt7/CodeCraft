import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv('assets/learn.csv')

print(data.isnull().sum())

X = data['code']
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    keras.layers.LSTM(units=64),
    keras.layers.Dense(units=len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

def predict_error(code):
    sequence = tokenizer.texts_to_sequences([code])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    if np.any(np.isnan(padded_sequence)):
        return "Input data contains NaN values."

    prediction = model.predict(padded_sequence)
    predicted_label = prediction.argmax()
    return predicted_label

new_code = "#include <iostream>\nint main() {\nint a = 5;\nint b = 0;\nstd::cout << a / b << std::endl;\nreturn 0;\n}"
print(f"Predicted label (code with error): {predict_error(new_code)}")
new_bad_code = "#include <iostream>\nint main(){\n int a"
print(f"Predicted label (incomplete code): {predict_error(new_bad_code)}")