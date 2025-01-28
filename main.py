import numpy as np
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

loaded_model = keras.models.load_model('cpp_err.h5')

with open('tokenizer_cpp_err.pkl', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

def predict_error(code):
    sequence = loaded_tokenizer.texts_to_sequences([code])

    max_length = loaded_model.input_shape[1] # get max length
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    prediction = loaded_model.predict(padded_sequence)

    return int(prediction.round())


new_code = "#include <iostream>\nint main() {\nint b = 10;\nb = b / 5;\nstd::cout << b << std::endl;\nreturn 0;\n}"
print(f"Predicted label (no error): {predict_error(new_code)}")

new_logic_code = "#include <iostream>\nint main(){\nint a = 10;\nif(a=10){\nstd::cout<< a;\nreturn 0;\n}"
print(f"Predicted label (error): {predict_error(new_logic_code)}")