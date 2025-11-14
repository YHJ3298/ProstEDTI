import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split

# --------------------------
# Define paths for final result folders (only save final files)
# --------------------------
final_drug_folder = "final_drug_smi"  # Final drug .smi files
final_target_folder = "final_target_fasta"  # Final target FASTA.txt files

# Create final result folders
for folder in [final_drug_folder, final_target_folder]:
    os.makedirs(folder, exist_ok=True)

# --------------------------
# 1. Read and merge original datasets (processed in memory, no intermediate files saved)
# --------------------------
df1 = pd.read_csv('./BindingDB/test.csv')
df2 = pd.read_csv('./BindingDB/train.csv')
df3 = pd.read_csv('./BindingDB/val.csv')

required_cols = ['Unnamed: 0', 'SMILES', 'Target Sequence', 'Label']
cols = list(df1.columns)
if (list(df2.columns) == cols and list(df3.columns) == cols
        and all(col in cols for col in required_cols)):

    # Merge data (in memory only, no intermediate save)
    combined_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    combined_df['Unnamed: 0'] = range(len(combined_df))
    print("1. Dataset merging completed (processed in memory, no intermediate files saved)")

    # --------------------------
    # 2. Separate positive and negative samples (processed in memory, no intermediate files saved)
    # --------------------------
    positive_samples = combined_df[combined_df['Label'] == 1].copy()
    negative_samples = combined_df[combined_df['Label'] == 0].copy()
    positive_samples['Unnamed: 0'] = range(len(positive_samples))
    negative_samples['Unnamed: 0'] = range(len(negative_samples))
    print(f"2. Positive/negative sample separation completed ({len(positive_samples)} positive samples, {len(negative_samples)} negative samples; no intermediate files saved)")

    # --------------------------
    # 3. Split into training and test sets (processed in memory, no intermediate files saved)
    # --------------------------
    # Split positive samples
    pos_train, pos_test = train_test_split(positive_samples, test_size=0.2, random_state=42)
    pos_train['Unnamed: 0'] = range(len(pos_train))
    pos_test['Unnamed: 0'] = range(len(pos_test))

    # Split negative samples
    neg_train, neg_test = train_test_split(negative_samples, test_size=0.2, random_state=42)
    neg_train['Unnamed: 0'] = range(len(neg_train))
    neg_test['Unnamed: 0'] = range(len(neg_test))
    print("3. Training/test set splitting completed (processed in memory, no intermediate files saved)")

    # --------------------------
    # 4. Extract core drug and target data (retain only necessary columns, no intermediate CSVs saved)
    # --------------------------
    drug_cols = ['Unnamed: 0', 'SMILES', 'Label']  # Necessary columns for drugs
    target_cols = ['Unnamed: 0', 'Target Sequence', 'Label']  # Necessary columns for targets

    # Drug data (temporarily stored in memory)
    drug_data = {
        'positive_train': pos_train[drug_cols],
        'positive_test': pos_test[drug_cols],
        'negative_train': neg_train[drug_cols],
        'negative_test': neg_test[drug_cols]
    }

    # Target data (temporarily stored in memory)
    target_data = {
        'positive_train': pos_train[target_cols],
        'positive_test': pos_test[target_cols],
        'negative_train': neg_train[target_cols],
        'negative_test': neg_test[target_cols]
    }

    # --------------------------
    # 5. Generate final drug .smi files (only save this result)
    # Format: Smiles column first, ID column second (tab-separated)
    # --------------------------
    print("\n4. Generating final drug .smi files...")
    for prefix, df in drug_data.items():
        # Extract SMILES directly from in-memory data, no temporary files needed
        smi_df = pd.DataFrame({'Smiles': df['SMILES']})
        smi_df['ID'] = [f'molecule_{i + 1}' for i in range(len(smi_df))]

        # Save final .smi files (only this step is saved)
        final_smi_path = os.path.join(final_drug_folder, f'{prefix}_drug.smi')
        smi_df[['Smiles', 'ID']].to_csv(final_smi_path, sep='\t', index=False)
        print(f"   Saved: {final_smi_path}")

    # --------------------------
    # 6. Generate final target FASTA.txt files (only save this result)
    # Format: >P{index}\n{sequence}
    # --------------------------
    print("\n5. Generating final target FASTA.txt files...")
    for prefix, df in target_data.items():
        # Write FASTA directly from in-memory data, no intermediate CSVs needed
        final_fasta_path = os.path.join(final_target_folder, f'{prefix}_target.txt')

        with open(final_fasta_path, 'w') as fasta_out:
            for _, row in df.iterrows():  # Iterate over in-memory dataframe
                index = row['Unnamed: 0']  # Index column
                sequence = row['Target Sequence']  # Target sequence
                fasta_out.write(f'>P{index}\n{sequence}\n')

        print(f"   Saved: {final_fasta_path}")

    print("\nAll processing completed! Only final result files are retained:")
    print(f"   Drug .smi files path: {final_drug_folder}")
    print(f"   Target FASTA files path: {final_target_folder}")

else:
    missing_cols = [col for col in required_cols if col not in cols]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols} (please confirm column names are 'SMILES', 'Target Sequence', 'Label')")
    else:
        print("Error: Column names/order inconsistent across the three datasets")