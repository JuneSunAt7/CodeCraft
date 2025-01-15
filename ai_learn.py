import numpy as np
import json

with open('../assets/parsed_before.json', 'r') as json_file:
    before_data = json.load(json_file)

with open('../assets/parsed_after.json', 'r') as json_file:
    after_data = json.load(json_file)

X = []
y = []

def extract_features(data):
    features = []
    for item in data:
        features.append([len(item['name']), len(item['kind'])])  # Простой набор признаков
    return features

X_before = extract_features(before_data)
X_after = extract_features(after_data)

X.extend(X_before)
X.extend(X_after)

y.extend([0] * len(X_before))
y.extend([1] * len(X_after))

X = np.array(X)
y = np.array(y)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=5, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)