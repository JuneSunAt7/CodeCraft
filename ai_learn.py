import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Загрузка данных
data = pd.read_csv('assets/train.csv')

# Проверка меток
print(data['label'].value_counts())

X = data['code']
y = data['label']

# Кодирование меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Токенизация и последовательности
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Определение максимальной длины последовательности
max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Создание модели
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    keras.layers.LSTM(units=128, return_sequences=True),
    keras.layers.LSTM(units=64),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=len(label_encoder.classes_), activation='softmax')  # Используем softmax для многоклассовой классификации
])

# Компиляция модели
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Убедитесь, что Learning Rate адекватен
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=2)  # Увеличиваем количество эпох

# Оценка модели
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# Функция предсказания
def predict_error(code):
    sequence = tokenizer.texts_to_sequences([code])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_label = prediction.argmax()
    return label_encoder.inverse_transform([predicted_label])[0]  # Декодируем обратно метку

# Пример использования
new_code = "#include <iostream>\nint main() {\nint a = 5;\nint b = 10;\nstd::cout << a + b << std::endl;\nreturn 0;\n}"
print(f"Predicted label (no error): {predict_error(new_code)}")

new_bad_code = "#include <iostream>\nint main(){\n int a}"  # Пример синтаксической ошибки
print(f"Predicted label (syntax error): {predict_error(new_bad_code)}")

new_logic_code = "#include <iostream>\nint main() {\nint a = 5;\nstd::cout << a / 0 << std::endl;\nreturn 0;\n}"  # Пример логической ошибки
print(f"Predicted label (logic error): {predict_error(new_logic_code)}")