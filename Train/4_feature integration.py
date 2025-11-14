import pandas as pd


# Define a function to process files
def process_files(drug_file, target_file, label):
    # Read drug features and target features
    drug_df = pd.read_csv(drug_file)
    target_df = pd.read_csv(target_file)

    # Check if the number of rows is consistent
    if len(drug_df) != len(target_df):
        raise ValueError(f"The number of rows in {drug_file} and {target_file} is inconsistent, please check the input files.")

    # Concatenate drug features and target features
    merged_df = pd.concat([drug_df, target_df], axis=1)

    # Add label column
    merged_df['label'] = label

    return merged_df


# Define a function to merge and save files (directly accept DataFrames as parameters)
def merge_and_save(train_pos_df, train_neg_df, test_pos_df, test_neg_df):
    # Concatenate positive and negative training sets (positive samples above negative samples)
    train_df = pd.concat([train_pos_df, train_neg_df], axis=0, ignore_index=True)

    # Concatenate positive and negative test sets (positive samples above negative samples)
    test_df = pd.concat([test_pos_df, test_neg_df], axis=0, ignore_index=True)

    # Save merged training and test sets
    train_df.to_csv('mol2vec_ProstT5_train.csv', index=False)
    test_df.to_csv('mol2vec_ProstT5_test.csv', index=False)

    print("Files 'mol2vec_ProstT5_train.csv' and 'mol2vec_ProstT5_test.csv' have been saved successfully.")


# Define the correspondence between filenames and labels
files_info = [
    ('negative_train_drug.csv', 'negative_train_target.csv', 0),
    ('negative_test_drug.csv', 'negative_test_target.csv', 0),
    ('positive_train_drug.csv', 'positive_train_target.csv', 1),
    ('positive_test_drug.csv', 'positive_test_target.csv', 1)
]

# Used to store processed DataFrames (no intermediate files saved)
data_dict = {}

# Iterate through file information, process each pair of files and store in dictionary
for drug_file, target_file, label in files_info:
    # Process each pair of files
    result_df = process_files(drug_file, target_file, label)

    # Extract data type from filename (training/test & positive/negative samples)
    parts = drug_file.split('_')
    data_type = f"{parts[1]}_{parts[0]}"  # Generate keys like 'train_negative', 'test_negative', etc.
    data_dict[data_type] = result_df

# Call the merge function to process training and test sets (use in-memory DataFrames directly)
merge_and_save(
    data_dict['train_positive'],  # Positive training set
    data_dict['train_negative'],  # Negative training set
    data_dict['test_positive'],  # Positive test set
    data_dict['test_negative']  # Negative test set
)