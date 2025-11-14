import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
import numpy as np

# Define function to read CSV file and perform EditedNearestNeighbours undersampling
def UnderENN_CSV(input_file, output_file, label_column, sampling_strategy='auto'):
    """
    Read data from CSV file, perform EditedNearestNeighbours undersampling, and save the result.

    Parameters:
    - input_file: Path to input CSV file
    - output_file: Path to save undersampled CSV file
    - label_column: Name of the label column
    - sampling_strategy: Undersampling strategy ('auto' or 'all')
    """
    # Read data
    df = pd.read_csv(input_file)

    # Extract feature columns and label column
    X = df.drop(label_column, axis=1)  # Features
    y = df[label_column]  # Label

    # Display number of each label before undersampling
    print("Number of each label before undersampling:")
    print(y.value_counts())

    # Extract positive and negative samples
    positive_samples = df[df[label_column] == 1]
    negative_samples = df[df[label_column] == 0]

    # Get the number of positive samples
    num_positive = len(positive_samples)

    # Undersample negative samples to match the number of positive samples
    negative_samples_resampled = negative_samples.sample(n=num_positive, random_state=42)

    # Combine positive samples and resampled negative samples
    balanced_data = pd.concat([positive_samples, negative_samples_resampled])

    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Display number of each label after undersampling
    print("\nNumber of each label after undersampling:")
    print(Counter(balanced_data[label_column]))

    # Save the undersampled dataset
    balanced_data.to_csv(output_file, index=False)
    print(f"Undersampling completed, saved as '{output_file}'.")


# Example usage
if __name__ == "__main__":
    # Input file path
    input_csv = 'mol2vec_ProstT5_train.csv'
    output_csv = 'mol2vec_ProstT5_train_enn.csv'

    # Specify label column name
    label_col = 'label'

    # Call function for undersampling
    UnderENN_CSV(input_csv, output_csv, label_col)