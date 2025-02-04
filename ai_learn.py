import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle
import numpy as np

data = pd.read_csv('assets/err_classificator.csv')

print(data.isnull().sum())
data.dropna(inplace=True)

X = data['code']
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_encoded, test_size=0.2, stratify=y, random_state=42
)

model = keras.Sequential([
    keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=128,
        input_length=max_length
    ),
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

model.save('cpp_error_multiclass.h5')
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

def predict_error(code):
    sequence = tokenizer.texts_to_sequences([code])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    class_idx = np.argmax(prediction)
    return label_encoder.inverse_transform([class_idx])[0]

test_codes = [
    "#include <iostream>\nint main() { std::cout << 10/0; }",  # division_by_zero
    "#include <iostream>\nint main() { int* p; *p = 5; }",      # null_pointer
    "int main() { return 0; }",                                 # syntax_error
    "#include <iostream>\nint main() { int x = 5; x++; }"       # no_error
]

for code in test_codes:
    print(f"Code:\n{code}\nPrediction: {predict_error(code)}\n{'-'*50}")