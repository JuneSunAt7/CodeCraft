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
# null is bad code one is good code(after-bef. refactor)
y.extend([0] * len(X_before))
y.extend([1] * len(X_after))

X = np.array(X)
y = np.array(y)