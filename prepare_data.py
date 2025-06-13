import itertools
import numpy as np
import pandas as pd

def generate_inputs():
    inputs = []

    # Generate all 2-hot combinations for first 16 bits (C(16, 2) = 120)
    first_combos = list(itertools.combinations(range(16), 2))

    # Generate all 1-hot combinations for second 16 bits (16 combinations)
    second_combos = [np.eye(16, dtype=int)[i] for i in range(16)]

    for first_pos in first_combos:
        first_part = np.zeros(16, dtype=int)
        first_part[list(first_pos)] = 1
        for second_part in second_combos:
            full_input = np.concatenate([first_part, second_part])
            inputs.append(full_input)

    return np.array(inputs)  # shape: (1920, 32)

def read_labels_from_excel(excel_path):
    # Read Excel and flatten in row-major order (left-to-right, top-to-bottom)
    df = pd.read_excel(excel_path, header=None)
    return df.to_numpy().flatten(order='C')  # shape: (19200,)

def get_label_set(flat_labels, set_index):
    start = set_index * 1920
    end = start + 1920
    return flat_labels[start:end]

def save_dataset(inputs, labels, filename):
    assert len(inputs) == len(labels), "Inputs and labels length mismatch"
    df = pd.DataFrame(inputs)
    df['label'] = labels
    df.to_csv(filename, index=False, header=False)

if __name__ == "__main__":
    # === Change this to your actual Excel file path ===
    excel_file = "label_data.xlsx"

    inputs = generate_inputs()
    flat_labels = read_labels_from_excel(excel_file)

    # Save training set (label set 0)
    train_labels = get_label_set(flat_labels, set_index=0)
    save_dataset(inputs, train_labels, filename="train_set.csv")
    print("Saved train_set.csv")

    # 4. Save remaining test sets (label sets 1 to 9)
    for i in range(1, 10):
        test_labels = get_label_set(flat_labels, set_index=i)
        filename = f"test_set_{i}.csv"
        save_dataset(inputs, test_labels, filename)
        print(f"Saved {filename}")

