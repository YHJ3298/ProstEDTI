import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec


def extract_mol2vec_features(
        input_file,
        model_path,
        output_file,
        radius=1,
        uncommon_token="UNK"
):
    """
    Extract mol2vec features from molecular files
    """
    # Step 1: Load molecular data
    print("Loading molecular data...")
    file_ext = input_file.split('.')[-1].lower()
    if file_ext in ['smi', 'ism']:
        # Read SMI file (Fixed parameter: use on_bad_lines instead of error_bad_lines)
        df = pd.read_csv(
            input_file,
            delimiter='\t',
            usecols=[0, 1],
            names=['Smiles', 'ID'],
            on_bad_lines='skip'  # Skip bad lines (replaces the original error_bad_lines=False)
        )
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol='Smiles')
    elif file_ext == 'sdf':
        df = PandasTools.LoadSDF(input_file)
        df['Smiles'] = df['ROMol'].apply(Chem.MolToSmiles)
    else:
        raise ValueError("Unsupported file format, please use SMI or SDF")

    # Filter invalid molecules
    df = df[df['ROMol'].notnull()].reset_index(drop=True)
    print(f"Successfully loaded {len(df)} valid molecules")

    # Step 2: Generate molecular substructure sequences
    print("Generating molecular substructure sequences...")
    df['mol_sentence'] = df['ROMol'].apply(
        lambda mol: mol2alt_sentence(mol, radius)
    )

    # Step 3: Load model and generate feature vectors
    print("Generating mol2vec features...")
    model = word2vec.Word2Vec.load(model_path)

    vectors = sentences2vec(
        df['mol_sentence'],
        model,
        unseen=uncommon_token
    )

    # Save results (remove ID and Smiles columns, retain only mol2vec features)
    feat_cols = [f"mol2vec_{i:03d}" for i in range(vectors.shape[1])]
    feat_df = pd.DataFrame(vectors, columns=feat_cols)
    feat_df.to_csv(output_file, index=False)  # Save only feature vectors, excluding ID and Smiles columns
    print(f"Feature extraction completed, saved to {output_file}")


if __name__ == "__main__":
    # List of input file paths
    input_files = [
        "./final_drug_smi/negative_train_drug.smi",
        "./final_drug_smi/negative_test_drug.smi",
        "./final_drug_smi/positive_train_drug.smi",
        "./final_drug_smi/positive_test_drug.smi"
    ]

    # List of output file paths
    output_files = [
        "negative_train_drug.csv",
        "negative_test_drug.csv",
        "positive_train_drug.csv",
        "positive_test_drug.csv"
    ]

    # Process each file iteratively
    for input_file, output_file in zip(input_files, output_files):
        extract_mol2vec_features(
            input_file=input_file,
            model_path="mol2vec/model.pkl",  # Ensure this is your model path
            output_file=output_file,
            radius=1,
            uncommon_token="UNK"
        )