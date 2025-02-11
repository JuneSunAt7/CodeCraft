import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

def normalize_code(code):
    return ' '.join(code.replace("\n", " ").replace("\t", " ").strip().split())

# Load the balanced and augmented dataset
data = pd.read_csv('assets/large_balanced_dataset.csv')
data.dropna(inplace=True)
data['code'] = data['code'].apply(normalize_code)

# Preprocess the data
X = data['code']
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_encoded, test_size=0.2, stratify=y, random_state=42
)

# Compute class weights (optional for balanced datasets)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    keras.layers.Conv1D(256, 5, activation='relu'),  # CNN layer for local pattern recognition
    keras.layers.GlobalMaxPooling1D(),                # Pooling layer
    keras.layers.Dense(128, activation='relu'),        # Dense layer
    keras.layers.Dropout(0.3),                       # Dropout for regularization
    keras.layers.Dense(num_classes, activation='softmax')  # Output layer
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Adjusted learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=100,  # Allow more epochs before stopping
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,  # Smaller batch size for better generalization
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict,
    verbose=2
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

y_pred = np.argmax(model.predict(X_test), axis=1)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

model.save('cpp_error_multiclass.h5')

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

def predict_error(code):
    if not isinstance(code, str) or len(code.strip()) == 0:
        return "Invalid input"
    sequence = tokenizer.texts_to_sequences([normalize_code(code)])
    if not sequence or len(sequence[0]) == 0:
        return "Invalid input"
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    print(f"Code:\n{code}\nPrediction probabilities: {prediction}")
    class_idx = np.argmax(prediction)
    return label_encoder.inverse_transform([class_idx])[0]

test_codes = [
    "#include <iostream>\nint main() { std::cout << 10/0; }",  # division_by_zero
    "#include <iostream>\nint main() { int* p = nullptr; *p = 5; }",  # null_pointer
    "int main() { return 0 }",  # syntax_error
    "#include <iostream>\nint main() { int x = 5; x++; return 0; }"  # no_error
]

for code in test_codes:
    print(f"Code:\n{code}\nPrediction: {predict_error(code)}\n{'-'*50}")