import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle

data = pd.read_csv('assets/train.csv')

print(data.isnull().sum())

data.dropna(inplace=True)
X = data['code']
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, stratify=y, random_state=42)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(units=64)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

model.save('cpp_err.h5')

with open('tokenizer_cpp_err.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def predict_error(code):
    sequence = tokenizer.texts_to_sequences([code])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    return int(prediction.round())

new_code = "#include <iostream>\nint main() {\nstd::cout << \"Welcome!\";\nreturn 0;\n}"
print(f"Predicted label (no error): {predict_error(new_code)}")

new_error_code = "#include <iostream>\nint main() {\nint a = 10;\na = a / 0;\nstd::cout << a << std::endl;\nreturn 0;\n}"
print(f"Predicted label (error): {predict_error(new_error_code)}")

new_logic_code = "#include <iostream>\nint main(){\nint a = 10;\nif(a=10){\nstd::cout<< a;\nreturn 0;\n}"
print(f"Predicted label (error): {predict_error(new_logic_code)}")