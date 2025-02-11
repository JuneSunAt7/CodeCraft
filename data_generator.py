import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('assets/balanced_dataset_300.csv')

# Check label distribution
label_counts = data['label'].value_counts()
print("Original Label Distribution:")
print(label_counts)
total_samples = 1200
samples_per_class = total_samples // len(label_counts)  # Equal distribution
print(f"Desired samples per class: {samples_per_class}")
balanced_data = pd.DataFrame()

for label in label_counts.index:
    class_data = data[data['label'] == label]
    sampled_data = class_data.sample(n=samples_per_class, random_state=42, replace=True)
    balanced_data = pd.concat([balanced_data, sampled_data], ignore_index=True)

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify the new label distribution
print("Balanced Label Distribution:")
print(balanced_data['label'].value_counts())

def inject_logic_error(code):
    if "return 0;" in code:
        return code.replace("return 0;", "std::cout << 10 / 0; return 0;")
    return code

def inject_runtime_error(code):
    if "int main()" in code:
        return code.replace("int main()", "int main() { int* ptr = nullptr; std::cout << *ptr;")
    return code

def inject_syntax_error(code):
    if "return 0;" in code:
        return code.replace("return 0;", "return 0")
    return code

def augment_data(data, factor=2):
    augmented_data = data.copy()
    for _ in range(factor - 1):  # Repeat augmentation based on the factor
        new_data = data.copy()
        new_data['code'] = new_data['code'].apply(lambda x: inject_logic_error(x) if np.random.rand() < 0.33 else (
                                                  inject_runtime_error(x) if np.random.rand() < 0.5 else inject_syntax_error(x)))
        augmented_data = pd.concat([augmented_data, new_data], ignore_index=True)
    return augmented_data

# Augment the dataset
augmented_data = augment_data(data, factor=2)  # Double the size
print("Augmented Label Distribution:")
print(augmented_data['label'].value_counts())

augmented_data.to_csv('assets/large_balanced_dataset.csv', index=False)
print("Balanced and augmented dataset saved to 'assets/large_balanced_dataset.csv'")