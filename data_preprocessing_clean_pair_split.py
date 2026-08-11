# -*- coding: utf-8 -*-
"""
data_preprocessing_clean_pair_split.py

Function
--------
Starting from merging the original BindingDB train/test/val CSV files, first perform pair-level data cleaning, then re-split the data into training and test sets at an 8:2 ratio,
and finally output the .smi and .fasta files required for subsequent mol2vec / ProstT5 feature extraction.

Compared with the original data preprocessing.py, this script adds:
1. Missing-value handling: remove samples with missing SMILES / Target Sequence / Label;
2. SMILES standardization: use RDKit to generate canonical SMILES by default;
3. Target Sequence standardization: remove spaces, convert to uppercase, and replace U/Z/O/B with X;
4. Duplicate handling:
   - Do not remove individually duplicated drugs;
   - Do not remove individually duplicated targets;
   - Handle duplicates only at the drug-target pair level;
   - Keep only one record for completely duplicated drug-target-label triplets;
   - By default, remove all samples with the same drug-target pair but conflicting labels;
5. Re-splitting:
   - By default, perform a label-stratified 8:2 split on all cleaned data;
   - Optionally force the original Train=26073, Test=6520 sizes, provided that enough samples remain after cleaning.
6. Output validation reports:
   - clean_bindingdb_pair_table.csv
   - removed_records.csv
   - preprocessing_summary.json / csv

Default input:
./BindingDB/train.csv
./BindingDB/test.csv
./BindingDB/val.csv

Default output:
./final_drug_smi/
./final_target_fasta/
./preprocessing_report/

Default final files:
./final_drug_smi/positive_train_drug.smi
./final_drug_smi/positive_test_drug.smi
./final_drug_smi/negative_train_drug.smi
./final_drug_smi/negative_test_drug.smi

./final_target_fasta/positive_train_target.fasta
./final_target_fasta/positive_test_target.fasta
./final_target_fasta/negative_train_target.fasta
./final_target_fasta/negative_test_target.fasta

Run:
python data_preprocessing_clean_pair_split.py

If you want to try forcing the original sample sizes:
python data_preprocessing_clean_pair_split.py --preserve-original-size true

Note:
If fewer than 32593 samples remain after cleaning, --preserve-original-size true will raise an error directly.
In this case, Train=26073 and Test=6520 cannot be forcibly preserved unless you have a larger original BindingDB sample pool.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLS = ["Unnamed: 0", "SMILES", "Target Sequence", "Label"]

DEFAULT_INPUT_FILES = [
    "./BindingDB/train.csv",
    "./BindingDB/test.csv",
    "./BindingDB/val.csv",
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ["true", "1", "yes", "y"]:
        return True
    if v in ["false", "0", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError("Boolean expected: true/false")


def safe_id(text: str, fallback: str) -> str:
    text = str(text).strip()
    if not text:
        text = fallback
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.|:-]+", "_", text)
    return text[:180]


def check_required_columns(df: pd.DataFrame, file_path: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path} is missing required columns: {missing}\n"
            f"Current columns: {list(df.columns)}"
        )


def load_bindingdb_files(input_files):
    dfs = []
    for fp in input_files:
        if not Path(fp).exists():
            raise FileNotFoundError(f"Input file not found: {fp}")
        df = pd.read_csv(fp)
        check_required_columns(df, fp)
        df = df[REQUIRED_COLS].copy()
        df["source_file"] = Path(fp).name
        df["source_row"] = range(len(df))
        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    combined["original_global_index"] = range(len(combined))
    return combined


def canonicalize_smiles(smiles: str, canonicalize: bool = True):
    smiles = str(smiles).strip()
    if smiles == "" or smiles.lower() in ["nan", "none", "null"]:
        return None, "missing_smiles"

    if not canonicalize:
        return smiles, None

    try:
        from rdkit import Chem
    except Exception:
        # If RDKit is not available in the environment, keep the original SMILES but report that canonicalization cannot be performed
        return smiles, "rdkit_not_available_no_canonicalization"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    return Chem.MolToSmiles(mol, canonical=True), None


def normalize_target_sequence(seq: str):
    seq = str(seq).replace(" ", "").replace("\t", "").replace("\n", "").upper()
    if seq == "" or seq.lower() in ["nan", "none", "null"]:
        return None, "missing_target_sequence"

    # Common ProstT5 / ProtT5 processing: replace non-standard amino acids with X
    for ch in ["U", "Z", "O", "B"]:
        seq = seq.replace(ch, "X")

    return seq, None


def normalize_label(label):
    if pd.isna(label):
        return None, "missing_label"

    try:
        value = int(label)
    except Exception:
        return None, "invalid_label"

    if value not in [0, 1]:
        return None, "invalid_label"

    return value, None


def preprocess_basic(df: pd.DataFrame, canonicalize_smiles_flag=True, invalid_smiles_action="error"):
    """
    Basic cleaning:
    - Missing SMILES / target / label;
    - Invalid SMILES;
    - Target sequence standardization;
    - Label standardization.
    """
    keep_rows = []
    removed_rows = []

    for _, row in df.iterrows():
        can_smiles, smiles_error = canonicalize_smiles(
            row["SMILES"],
            canonicalize=canonicalize_smiles_flag
        )
        norm_target, target_error = normalize_target_sequence(row["Target Sequence"])
        norm_label, label_error = normalize_label(row["Label"])

        new_row = row.to_dict()
        new_row["drug_smiles"] = can_smiles
        new_row["target_sequence"] = norm_target
        new_row["label"] = norm_label
        new_row["smiles_error"] = smiles_error
        new_row["target_error"] = target_error
        new_row["label_error"] = label_error

        errors = [e for e in [smiles_error, target_error, label_error] if e is not None]

        # RDKit unavailability is not treated as a reason for removal; it is only reported
        errors_to_remove = [e for e in errors if e != "rdkit_not_available_no_canonicalization"]

        if errors_to_remove:
            reason = ";".join(errors_to_remove)
            if "invalid_smiles" in errors_to_remove and invalid_smiles_action == "error":
                raise ValueError(
                    f"Invalid SMILES detected; terminating by default.\n"
                    f"source_file={row.get('source_file')}, source_row={row.get('source_row')}, SMILES={row.get('SMILES')}\n"
                    f"To remove invalid SMILES and continue running, set --invalid-smiles-action drop"
                )
            new_row["remove_reason"] = reason
            removed_rows.append(new_row)
            continue

        new_row["pair_key"] = f"{can_smiles}||{norm_target}"
        new_row["triplet_key"] = f"{can_smiles}||{norm_target}||{norm_label}"
        keep_rows.append(new_row)

    keep_df = pd.DataFrame(keep_rows)
    removed_df = pd.DataFrame(removed_rows)

    if keep_df.empty:
        raise ValueError("No data remains after basic cleaning; please check the input files.")

    return keep_df, removed_df


def remove_duplicates_and_conflicts(df: pd.DataFrame, removed_df: pd.DataFrame, conflict_action="remove_all"):
    """
    Pair-level deduplication:
    1. Same drug-target pair with conflicting labels: remove all by default;
    2. Completely duplicated drug-target-label triplets: keep only the first record.
    """
    work = df.copy()
    removed_list = []
    if removed_df is not None and not removed_df.empty:
        removed_list.append(removed_df.copy())

    # 1. Pairs with conflicting labels
    label_nunique = work.groupby("pair_key")["label"].nunique()
    conflict_pairs = set(label_nunique[label_nunique > 1].index)

    if conflict_pairs:
        conflict_rows = work[work["pair_key"].isin(conflict_pairs)].copy()
        conflict_rows["remove_reason"] = "conflicting_labels_same_pair"

        if conflict_action == "error":
            conflict_rows.to_csv("conflicting_label_pairs_debug.csv", index=False, encoding="utf-8-sig")
            raise ValueError(
                f"Found {len(conflict_pairs)} identical drug-target pairs with conflicting labels."
                f"Details have been saved to conflicting_label_pairs_debug.csv"
            )
        elif conflict_action == "remove_all":
            removed_list.append(conflict_rows)
            work = work[~work["pair_key"].isin(conflict_pairs)].copy()
        elif conflict_action == "keep_first":
            # Not recommended; for debugging only
            work = work.sort_values(["pair_key", "original_global_index"]).drop_duplicates("pair_key", keep="first").copy()
        else:
            raise ValueError(f"Unsupported conflict_action: {conflict_action}")

    # 2. Keep only the first record for completely duplicated triplets
    dup_mask = work.duplicated(subset=["triplet_key"], keep="first")
    if dup_mask.any():
        dup_rows = work[dup_mask].copy()
        dup_rows["remove_reason"] = "duplicate_triplet_keep_first"
        removed_list.append(dup_rows)
        work = work[~dup_mask].copy()

    removed_all = pd.concat(removed_list, axis=0, ignore_index=True) if removed_list else pd.DataFrame()
    work = work.reset_index(drop=True)

    info = {
        "conflict_pair_count": int(len(conflict_pairs)),
        "duplicate_triplet_removed_count": int(dup_mask.sum()),
        "removed_total_records": int(len(removed_all)),
        "clean_total_records": int(len(work)),
    }

    return work, removed_all, info


def stratified_split_clean_df(
    clean_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    preserve_original_size: bool = False,
    target_train_size: int = 26073,
    target_test_size: int = 6520,
):
    """
    Re-stratify and split train/test on the cleaned data.

    preserve_original_size=False:
        Use all cleaned samples and perform a stratified split according to test_size.
    preserve_original_size=True:
        First attempt to sample target_train_size + target_test_size records from the cleaned samples, then split them exactly.
        Raise an error if the total number of cleaned samples is insufficient.
    """
    df = clean_df.copy()

    if preserve_original_size:
        target_total = int(target_train_size + target_test_size)
        if len(df) < target_total:
            raise ValueError(
                f"Insufficient samples after cleaning; the original sizes cannot be preserved.\n"
                f"clean_total={len(df)}, required_total={target_total}\n"
                f"This indicates that, within the current original BindingDB pool, strict deduplication/conflict removal cannot yield "
                f"Train={target_train_size}, Test={target_test_size}。\n"
                f"Disable --preserve-original-size or provide a larger original BindingDB data pool."
            )

        # Stratified sampling by label from the cleaned unique pairs to obtain target_total
        df, _ = train_test_split(
            df,
            train_size=target_total,
            stratify=df["label"],
            random_state=random_state
        )

        # Use an absolute test_size count to obtain target_test_size exactly
        train_df, test_df = train_test_split(
            df,
            test_size=target_test_size,
            stratify=df["label"],
            random_state=random_state
        )

        if len(train_df) != target_train_size or len(test_df) != target_test_size:
            raise RuntimeError(
                f"Exact split failed: train={len(train_df)}, test={len(test_df)}, "
                f"expected train={target_train_size}, test={target_test_size}"
            )

    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=random_state
        )

    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["split"] = "train"
    test_df["split"] = "test"

    # Check train/test pair overlap
    train_pairs = set(train_df["pair_key"])
    test_pairs = set(test_df["pair_key"])
    overlap = train_pairs & test_pairs
    if overlap:
        raise RuntimeError(
            f"Train/test pair overlap still exists after re-splitting: {len(overlap)}."
            f"In theory, this should not occur after splitting unique cleaned pairs; please check the code."
        )

    return train_df, test_df


def write_smi(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            mol_id = safe_id(row.get("Unnamed: 0", ""), f"molecule_{i+1}")
            f.write(f"{row['drug_smiles']}\t{mol_id}\n")


def write_fasta(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            tid = safe_id(row.get("Unnamed: 0", ""), f"P{i+1}")
            f.write(f">P{tid}\n")
            seq = str(row["target_sequence"])
            for start in range(0, len(seq), 80):
                f.write(seq[start:start + 80] + "\n")


def write_final_files(train_df: pd.DataFrame, test_df: pd.DataFrame, final_drug_folder: Path, final_target_folder: Path):
    final_drug_folder.mkdir(parents=True, exist_ok=True)
    final_target_folder.mkdir(parents=True, exist_ok=True)

    split_map = {
        "positive_train": train_df[train_df["label"] == 1].copy(),
        "positive_test": test_df[test_df["label"] == 1].copy(),
        "negative_train": train_df[train_df["label"] == 0].copy(),
        "negative_test": test_df[test_df["label"] == 0].copy(),
    }

    for prefix, sub in split_map.items():
        sub = sub.reset_index(drop=True)

        smi_path = final_drug_folder / f"{prefix}_drug.smi"
        fasta_path = final_target_folder / f"{prefix}_target.fasta"

        write_smi(sub, smi_path)
        write_fasta(sub, fasta_path)

        print(f"Saved {prefix}: {len(sub)} pairs")
        print(f"  drug  -> {smi_path}")
        print(f"  target-> {fasta_path}")


def save_reports(
    combined_df: pd.DataFrame,
    basic_clean_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_dir: Path,
    duplicate_info: Dict,
    args,
):
    report_dir.mkdir(parents=True, exist_ok=True)

    basic_clean_df.to_csv(report_dir / "basic_clean_pair_table_before_duplicate_handling.csv", index=False, encoding="utf-8-sig")
    clean_df.to_csv(report_dir / "clean_unique_pair_table.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(report_dir / "final_train_pair_table.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(report_dir / "final_test_pair_table.csv", index=False, encoding="utf-8-sig")

    if removed_df is not None and not removed_df.empty:
        removed_df.to_csv(report_dir / "removed_records.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["remove_reason"]).to_csv(report_dir / "removed_records.csv", index=False, encoding="utf-8-sig")

    summary = {
        "original_total_records": int(len(combined_df)),
        "after_basic_clean_records": int(len(basic_clean_df)),
        "clean_unique_total_records": int(len(clean_df)),

        "final_train_records": int(len(train_df)),
        "final_test_records": int(len(test_df)),
        "final_total_records": int(len(train_df) + len(test_df)),

        "train_positive": int((train_df["label"] == 1).sum()),
        "train_negative": int((train_df["label"] == 0).sum()),
        "test_positive": int((test_df["label"] == 1).sum()),
        "test_negative": int((test_df["label"] == 0).sum()),

        "total_positive": int(((pd.concat([train_df, test_df])["label"]) == 1).sum()),
        "total_negative": int(((pd.concat([train_df, test_df])["label"]) == 0).sum()),

        "removed_records": int(len(removed_df)) if removed_df is not None else 0,
        "removed_by_reason": removed_df["remove_reason"].value_counts().to_dict() if removed_df is not None and not removed_df.empty and "remove_reason" in removed_df.columns else {},

        "preserve_original_size": bool(args.preserve_original_size),
        "target_train_size": int(args.target_train_size),
        "target_test_size": int(args.target_test_size),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
    }
    summary.update(duplicate_info)

    with open(report_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([summary]).to_csv(report_dir / "preprocessing_summary.csv", index=False, encoding="utf-8-sig")

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run(args):
    input_files = [args.train_csv, args.test_csv, args.val_csv]

    print("=" * 100)
    print("BindingDB pair-level preprocessing and 8:2 re-splitting")
    print("=" * 100)

    combined_df = load_bindingdb_files(input_files)
    print(f"Loaded original records: {len(combined_df)}")

    basic_clean_df, initial_removed_df = preprocess_basic(
        combined_df,
        canonicalize_smiles_flag=args.canonicalize_smiles,
        invalid_smiles_action=args.invalid_smiles_action,
    )
    print(f"After missing/invalid cleaning: {len(basic_clean_df)} records")

    clean_df, removed_df, duplicate_info = remove_duplicates_and_conflicts(
        basic_clean_df,
        initial_removed_df,
        conflict_action=args.conflict_action,
    )
    print(f"After pair-level duplicate/conflict handling: {len(clean_df)} records")

    train_df, test_df = stratified_split_clean_df(
        clean_df,
        test_size=args.test_size,
        random_state=args.random_state,
        preserve_original_size=args.preserve_original_size,
        target_train_size=args.target_train_size,
        target_test_size=args.target_test_size,
    )

    print("\nFinal split:")
    print(f"  train: {len(train_df)}")
    print(f"  test : {len(test_df)}")
    print(f"  train label counts: {train_df['label'].value_counts().to_dict()}")
    print(f"  test label counts : {test_df['label'].value_counts().to_dict()}")

    write_final_files(
        train_df=train_df,
        test_df=test_df,
        final_drug_folder=Path(args.final_drug_folder),
        final_target_folder=Path(args.final_target_folder),
    )

    save_reports(
        combined_df=combined_df,
        basic_clean_df=basic_clean_df,
        clean_df=clean_df,
        removed_df=removed_df,
        train_df=train_df,
        test_df=test_df,
        report_dir=Path(args.report_dir),
        duplicate_info=duplicate_info,
        args=args,
    )

    print("\nAll processing completed.")
    print(f"Drug .smi files path: {Path(args.final_drug_folder).resolve()}")
    print(f"Target FASTA files path: {Path(args.final_target_folder).resolve()}")
    print(f"Report path: {Path(args.report_dir).resolve()}")


def build_parser():
    p = argparse.ArgumentParser(description="BindingDB pair-level preprocessing and stratified 8:2 split.")

    p.add_argument("--train-csv", type=str, default="./BindingDB/train.csv")
    p.add_argument("--test-csv", type=str, default="./BindingDB/test.csv")
    p.add_argument("--val-csv", type=str, default="./BindingDB/val.csv")

    p.add_argument("--final-drug-folder", type=str, default="final_drug_smi")
    p.add_argument("--final-target-folder", type=str, default="final_target_fasta")
    p.add_argument("--report-dir", type=str, default="preprocessing_report")

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--canonicalize-smiles", type=str2bool, default=True)
    p.add_argument("--invalid-smiles-action", type=str, default="error", choices=["error", "drop"])
    p.add_argument("--conflict-action", type=str, default="remove_all", choices=["remove_all", "error", "keep_first"])

    p.add_argument(
        "--preserve-original-size",
        type=str2bool,
        default=False,
        help="Try to keep Train=26073 and Test=6520 if clean pool is large enough."
    )
    p.add_argument("--target-train-size", type=int, default=26073)
    p.add_argument("--target-test-size", type=int, default=6520)

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
