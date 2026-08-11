# -*- coding: utf-8 -*-
"""
integrated_mol2vec_ProstT5_feature_pipeline.py

Integrate three scripts into one complete pipeline:
1. Drug SMILES -> mol2vec features
2. Target FASTA -> ProstT5 features
3. Concatenate mol2vec + ProstT5 features to generate the final training and test sets

Default raw drug data paths:
./final_drug_smi/negative_train_drug.smi
./final_drug_smi/negative_test_drug.smi
./final_drug_smi/positive_train_drug.smi
./final_drug_smi/positive_test_drug.smi

Default raw target data paths:
./final_target_fasta/negative_train_target.fasta
./final_target_fasta/negative_test_target.fasta
./final_target_fasta/positive_train_target.fasta
./final_target_fasta/positive_test_target.fasta

Default intermediate feature outputs:
./feature_cache/negative_train_drug.csv
./feature_cache/negative_test_drug.csv
./feature_cache/positive_train_drug.csv
./feature_cache/positive_test_drug.csv

./feature_cache/negative_train_target.csv
./feature_cache/negative_test_target.csv
./feature_cache/positive_train_target.csv
./feature_cache/positive_test_target.csv

Default final outputs:
./mol2vec_ProstT5_train.csv
./mol2vec_ProstT5_test.csv

Run:
python integrated_mol2vec_ProstT5_feature_pipeline.py \
  --mol2vec-model ./mol2vec/model.pkl \
  --prostt5-model ./ProstT5 \
  --feature-output-dir ./feature_cache \
  --final-output-dir . \
  --target-batch-size 4 \
  --target-max-length 512

If intermediate features have already been generated and you only want to re-integrate them:
python integrated_mol2vec_ProstT5_feature_pipeline.py \
  --extract-drug false \
  --extract-target false
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DRUG_FILES = {
    "negative_train": "./final_drug_smi/negative_train_drug.smi",
    "negative_test": "./final_drug_smi/negative_test_drug.smi",
    "positive_train": "./final_drug_smi/positive_train_drug.smi",
    "positive_test": "./final_drug_smi/positive_test_drug.smi",
}

DEFAULT_TARGET_FILES = {
    "negative_train": "./final_target_fasta/negative_train_target.fasta",
    "negative_test": "./final_target_fasta/negative_test_target.fasta",
    "positive_train": "./final_target_fasta/positive_train_target.fasta",
    "positive_test": "./final_target_fasta/positive_test_target.fasta",
}

LABELS = {
    "negative_train": 0,
    "negative_test": 0,
    "positive_train": 1,
    "positive_test": 1,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ["true", "1", "yes", "y"]:
        return True
    if v in ["false", "0", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError("Boolean expected: true/false")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def check_file(path: str, name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return p


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def make_output_paths(feature_output_dir: Path):
    drug_outputs = {
        "negative_train": feature_output_dir / "negative_train_drug.csv",
        "negative_test": feature_output_dir / "negative_test_drug.csv",
        "positive_train": feature_output_dir / "positive_train_drug.csv",
        "positive_test": feature_output_dir / "positive_test_drug.csv",
    }
    target_outputs = {
        "negative_train": feature_output_dir / "negative_train_target.csv",
        "negative_test": feature_output_dir / "negative_test_target.csv",
        "positive_train": feature_output_dir / "positive_train_target.csv",
        "positive_test": feature_output_dir / "positive_test_target.csv",
    }
    return drug_outputs, target_outputs


# =========================
# 1. Drug mol2vec feature extraction
# =========================

def read_smi_file(smi_file: str) -> pd.DataFrame:
    """
    Read a .smi file.
    By default, use the first field of each line as SMILES and the second field as ID.
    If no ID is present, automatically generate an ID.
    """
    rows = []
    with open(smi_file, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # If the first line looks like a header, skip it
            low = line.lower()
            if line_no == 1 and ("smiles" in low or "drug" in low):
                continue

            parts = line.replace("\t", " ").split()
            if len(parts) == 0:
                continue

            rows.append({
                "Smiles": parts[0].strip(),
                "ID": parts[1].strip() if len(parts) > 1 else f"mol_{line_no}",
                "source_line": line_no,
            })

    if len(rows) == 0:
        raise ValueError(f"The SMI file is empty or no valid SMILES could be parsed: {smi_file}")

    return pd.DataFrame(rows)


def extract_mol2vec_features(
    input_file: str,
    model_path: str,
    output_file: str,
    radius: int = 1,
    uncommon_token: str = "UNK",
    strict_valid_smiles: bool = True,
):
    """
    Extract mol2vec features from an SMI file.
    The output contains only mol2vec_000 ... mol2vec_xxx and does not save ID or SMILES.
    """
    print(f"[{now()}] [mol2vec] Input: {input_file}")
    print(f"[{now()}] [mol2vec] Output: {output_file}")

    try:
        from rdkit import Chem
        from mol2vec.features import mol2alt_sentence, sentences2vec
        from gensim.models import word2vec
    except Exception as e:
        raise ImportError(
            "mol2vec drug feature extraction requires rdkit, mol2vec, and gensim to be installed.\n"
            f"Original error: {repr(e)}"
        )

    check_file(input_file, "SMI input file")
    check_file(model_path, "mol2vec model file")

    df = read_smi_file(input_file)

    mols = []
    invalid_rows = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["Smiles"]))
        if mol is None:
            invalid_rows.append((idx, row["source_line"], row["Smiles"]))
        mols.append(mol)

    if invalid_rows:
        msg = "\n".join([f"idx={i}, line={line}, smiles={smi}" for i, line, smi in invalid_rows[:20]])
        if strict_valid_smiles:
            raise ValueError(
                f"Found {len(invalid_rows)} invalid SMILES. To avoid row misalignment between drugs and targets, execution is terminated by default.\n"
                f"First several invalid records:\n{msg}\n"
                f"If you really want to skip invalid SMILES, set --strict-valid-smiles false"
            )
        print(f"[WARN] Found {len(invalid_rows)} invalid SMILES; they have been skipped.")
        df["ROMol"] = mols
        df = df[df["ROMol"].notnull()].reset_index(drop=True)
    else:
        df["ROMol"] = mols

    print(f"[{now()}] [mol2vec] Valid molecule count: {len(df)}")

    print(f"[{now()}] [mol2vec] Generating molecular substructure sequences...")
    df["mol_sentence"] = df["ROMol"].apply(lambda mol: mol2alt_sentence(mol, radius))

    print(f"[{now()}] [mol2vec] Loading model: {model_path}")
    model = word2vec.Word2Vec.load(model_path)

    print(f"[{now()}] [mol2vec] Generating vectors...")
    vectors = sentences2vec(df["mol_sentence"], model, unseen=uncommon_token)

    feat_cols = [f"mol2vec_{i:03d}" for i in range(vectors.shape[1])]
    feat_df = pd.DataFrame(vectors, columns=feat_cols)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_file, index=False)
    print(f"[{now()}] [mol2vec] Completed: {output_file}, shape={feat_df.shape}")


# =========================
# 2. Target ProstT5 feature extraction
# =========================

def parse_fasta(file_path: str) -> List[Tuple[str, str]]:
    """
    Parse a FASTA file.
    Return [(protein_id, raw_sequence), ...]
    Note: the raw sequence without added spaces is returned here to avoid adding spaces repeatedly.
    """
    check_file(file_path, "FASTA input file")

    data = []
    protein_name = None
    seq_parts = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_name is not None:
                    seq = "".join(seq_parts).replace(" ", "").replace("\t", "")
                    if seq:
                        data.append((protein_name, seq))
                protein_name = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if protein_name is not None:
        seq = "".join(seq_parts).replace(" ", "").replace("\t", "")
        if seq:
            data.append((protein_name, seq))

    if len(data) == 0:
        raise ValueError(f"No sequences could be parsed from the FASTA file: {file_path}")

    return data


def format_protein_sequence(seq: str) -> str:
    """
    Common ProstT5/T5Tokenizer input format: add spaces between amino acids.
    Replace U/Z/O/B with X.
    """
    seq = str(seq).upper()
    for ch in ["U", "Z", "O", "B"]:
        seq = seq.replace(ch, "X")
    return " ".join(list(seq))


def load_prostt5_model(device, model_path: str):
    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except Exception as e:
        raise ImportError(
            "ProstT5 target feature extraction requires torch, transformers, and sentencepiece to be installed.\n"
            f"Original error: {repr(e)}"
        )

    check_file(model_path, "ProstT5 model directory/path")

    print(f"[{now()}] [ProstT5] Loading tokenizer: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    print(f"[{now()}] [ProstT5] Loading T5EncoderModel: {model_path}")
    model = T5EncoderModel.from_pretrained(model_path).to(device)
    model.eval()

    return model, tokenizer


def pool_embeddings(last_hidden_state, attention_mask, pooling: str = "first"):
    """
    pooling='first': preserve the original script logic and use outputs.last_hidden_state[:, 0, :]
    pooling='mean': perform mean pooling over non-padding tokens
    """
    if pooling == "first":
        return last_hidden_state[:, 0, :]

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    raise ValueError(f"Unsupported pooling: {pooling}. Use 'first' or 'mean'.")


def extract_prostt5_features(
    fasta_file: str,
    output_file: str,
    model,
    tokenizer,
    device,
    batch_size: int = 4,
    max_length: int = 512,
    pooling: str = "first",
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print(f"[{now()}] [ProstT5] Input: {fasta_file}")
    print(f"[{now()}] [ProstT5] Output: {output_file}")

    fasta_data = parse_fasta(fasta_file)
    sequences = [format_protein_sequence(seq) for _, seq in fasta_data]
    print(f"[{now()}] [ProstT5] Sequence count: {len(sequences)}")

    print(f"[{now()}] [ProstT5] Tokenizer encoding...")
    inputs = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )

    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    print(f"[{now()}] [ProstT5] Batch inference...")
    for batch_input_ids, batch_attention_mask in tqdm(loader, desc=f"ProstT5 {Path(fasta_file).name}", unit="batch"):
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            emb = pool_embeddings(outputs.last_hidden_state, batch_attention_mask, pooling=pooling)

        all_embeddings.append(emb.detach().cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    columns = [f"feature_{i}" for i in range(embeddings.shape[1])]
    feat_df = pd.DataFrame(embeddings, columns=columns)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_file, index=False)
    print(f"[{now()}] [ProstT5] Completed: {output_file}, shape={feat_df.shape}")


# =========================
# 3. Feature integration
# =========================

def process_pair(drug_file: str, target_file: str, label: int) -> pd.DataFrame:
    drug_df = pd.read_csv(drug_file)
    target_df = pd.read_csv(target_file)

    if len(drug_df) != len(target_df):
        raise ValueError(
            f"The number of rows in the drug features and target features does not match:\n"
            f"drug_file={drug_file}, rows={len(drug_df)}\n"
            f"target_file={target_file}, rows={len(target_df)}\n"
            f"Please check whether the original SMI/FASTA files correspond one-to-one and whether any invalid SMILES were skipped."
        )

    merged = pd.concat([drug_df, target_df], axis=1)
    merged["label"] = int(label)
    return merged


def integrate_features(
    drug_outputs: Dict[str, Path],
    target_outputs: Dict[str, Path],
    final_output_dir: Path,
    train_output_name: str = "mol2vec_ProstT5_train.csv",
    test_output_name: str = "mol2vec_ProstT5_test.csv",
):
    print_section("Step 3/3: Feature integration")

    data = {}
    for key in ["negative_train", "negative_test", "positive_train", "positive_test"]:
        label = LABELS[key]
        data[key] = process_pair(str(drug_outputs[key]), str(target_outputs[key]), label)
        print(f"[{now()}] merged {key}: shape={data[key].shape}, label={label}")

    train_df = pd.concat(
        [data["positive_train"], data["negative_train"]],
        axis=0,
        ignore_index=True
    )
    test_df = pd.concat(
        [data["positive_test"], data["negative_test"]],
        axis=0,
        ignore_index=True
    )

    final_output_dir.mkdir(parents=True, exist_ok=True)
    train_path = final_output_dir / train_output_name
    test_path = final_output_dir / test_output_name

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[{now()}] Saved training set: {train_path}, shape={train_df.shape}, label_counts={train_df['label'].value_counts().to_dict()}")
    print(f"[{now()}] Saved test set: {test_path}, shape={test_df.shape}, label_counts={test_df['label'].value_counts().to_dict()}")

    return train_path, test_path


# =========================
# Main pipeline
# =========================

def run(args):
    print_section("Integrated mol2vec + ProstT5 feature pipeline")
    print(f"[{now()}] Current working directory: {Path.cwd()}")
    print(f"[{now()}] feature_output_dir: {args.feature_output_dir}")
    print(f"[{now()}] final_output_dir: {args.final_output_dir}")

    feature_output_dir = Path(args.feature_output_dir)
    final_output_dir = Path(args.final_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    drug_outputs, target_outputs = make_output_paths(feature_output_dir)

    for k, v in DEFAULT_DRUG_FILES.items():
        check_file(v, f"drug input {k}")
    for k, v in DEFAULT_TARGET_FILES.items():
        check_file(v, f"target input {k}")

    if args.extract_drug:
        print_section("Step 1/3: Drug mol2vec feature extraction")
        for key in ["negative_train", "negative_test", "positive_train", "positive_test"]:
            input_file = DEFAULT_DRUG_FILES[key]
            output_file = drug_outputs[key]

            if args.skip_existing and output_file.exists():
                print(f"[{now()}] [mol2vec] Skipping existing file: {output_file}")
                continue

            extract_mol2vec_features(
                input_file=input_file,
                model_path=args.mol2vec_model,
                output_file=str(output_file),
                radius=args.mol2vec_radius,
                uncommon_token=args.mol2vec_uncommon_token,
                strict_valid_smiles=args.strict_valid_smiles,
            )
    else:
        print_section("Step 1/3: Skip drug mol2vec feature extraction")

    if args.extract_target:
        print_section("Step 2/3: Target ProstT5 feature extraction")
        try:
            import torch
        except Exception as e:
            raise ImportError(f"torch must be installed to run ProstT5 feature extraction. Original error: {repr(e)}")

        device = torch.device(
            args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[{now()}] [ProstT5] Using device: {device}")

        model, tokenizer = load_prostt5_model(device, args.prostt5_model)

        for key in ["negative_train", "negative_test", "positive_train", "positive_test"]:
            fasta_file = DEFAULT_TARGET_FILES[key]
            output_file = target_outputs[key]

            if args.skip_existing and output_file.exists():
                print(f"[{now()}] [ProstT5] Skipping existing file: {output_file}")
                continue

            extract_prostt5_features(
                fasta_file=fasta_file,
                output_file=str(output_file),
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.target_batch_size,
                max_length=args.target_max_length,
                pooling=args.target_pooling,
            )
    else:
        print_section("Step 2/3: Skip target ProstT5 feature extraction")

    train_path, test_path = integrate_features(
        drug_outputs=drug_outputs,
        target_outputs=target_outputs,
        final_output_dir=final_output_dir,
        train_output_name=args.train_output_name,
        test_output_name=args.test_output_name,
    )

    print_section("Pipeline finished")
    print(f"Final train: {train_path}")
    print(f"Final test : {test_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Integrated mol2vec + ProstT5 feature extraction and integration pipeline."
    )

    p.add_argument("--mol2vec-model", type=str, default="./mol2vec/model.pkl")
    p.add_argument("--prostt5-model", type=str, default="./ProstT5")

    p.add_argument("--feature-output-dir", type=str, default="./feature_cache")
    p.add_argument("--final-output-dir", type=str, default=".")
    p.add_argument("--train-output-name", type=str, default="mol2vec_ProstT5_train.csv")
    p.add_argument("--test-output-name", type=str, default="mol2vec_ProstT5_test.csv")

    p.add_argument("--extract-drug", type=str2bool, default=True)
    p.add_argument("--extract-target", type=str2bool, default=True)
    p.add_argument("--skip-existing", type=str2bool, default=True)

    p.add_argument("--mol2vec-radius", type=int, default=1)
    p.add_argument("--mol2vec-uncommon-token", type=str, default="UNK")
    p.add_argument(
        "--strict-valid-smiles",
        type=str2bool,
        default=True,
        help="true: invalid SMILES cause error to avoid row misalignment; false: skip invalid SMILES"
    )

    p.add_argument("--target-batch-size", type=int, default=4)
    p.add_argument("--target-max-length", type=int, default=512)
    p.add_argument("--target-pooling", type=str, default="first", choices=["first", "mean"])
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
