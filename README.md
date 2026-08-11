# ProstEDTI

**ProstEDTI** is a reproducible and interpretable framework for **drug–target interaction (DTI) prediction**. The revised pipeline combines rigorous pair-level data cleaning, Mol2vec drug representations, frozen ProstT5 target representations, nested SHAP feature selection, Pseudo-WkNNIR local priors, PU-inspired reliability weighting, ENN undersampling, and an equal-weight XGBoost–LightGBM soft-voting ensemble.

This README is aligned with the revised manuscript:

> **ProstEDTI: Achieving high-precision prediction of drug-target interactions using the advanced pre-trained architecture ProstT5 and the ENN undersampling algorithm**

**Authors:** Yun Zuo, Hongjin Yan, Xubin Wu, Fei Ge, Sirui Fei, Jingwen Liang, and Qiao Ning  
**Affiliation:** School of Artificial Intelligence and Computer Science, Jiangnan University, Wuxi, China  
**Repository:** https://github.com/YHJ3298/ProstEDTI  
**Web server:** www.prostedti.com

---

## Table of Contents

- [Overview](#overview)
- [Final Paper-Aligned Configuration](#final-paper-aligned-configuration)
- [Method](#method)
- [Dataset and Pair-Level Cleaning](#dataset-and-pair-level-cleaning)
- [Performance](#performance)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Pretrained ProstT5 Model](#pretrained-prostt5-model)
- [Quick Reproduction](#quick-reproduction)
- [Full Reproduction from Raw BindingDB Files](#full-reproduction-from-raw-bindingdb-files)
- [Output Files](#output-files)
- [Implementation Details](#implementation-details)
- [Reproducibility and Leakage Control](#reproducibility-and-leakage-control)
- [Interpretability and Case Study](#interpretability-and-case-study)
- [Important Notes](#important-notes)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

## Overview

Accurate DTI prediction is important for drug discovery, target identification, and drug repurposing. ProstEDTI was designed to address several practical problems that can otherwise lead to optimistic or unstable evaluation:

- duplicated drug–target pairs;
- conflicting labels for the same drug–target pair;
- high-dimensional and partially redundant protein embeddings;
- class imbalance and class-boundary noise;
- suspicious negative samples;
- information leakage during feature selection and preprocessing;
- limited interpretability of learned representations.

The revised ProstEDTI workflow uses:

1. **Global pair-level cleaning and stratified re-splitting** of BindingDB;
2. **64-dimensional Mol2vec** representations for drugs;
3. **1,024-dimensional frozen ProstT5** representations for targets;
4. **Nested SHAP Top320 selection** applied only to ProstT5 features;
5. **11-dimensional Pseudo-WkNNIR local-prior features**;
6. **PU-inspired continuous reliability weighting** for suspicious negative samples;
7. **Edited Nearest Neighbours (ENN)** with `k = 5`;
8. **XGBoost + LightGBM equal-weight soft voting**;
9. **Nested 10-fold cross-validation** for model assessment/configuration;
10. **One-time independent-test evaluation** after the final configuration is fixed.

The final model input has **395 dimensions**:

```text
64 Mol2vec
+ 320 SHAP-selected ProstT5
+ 11 Pseudo-WkNNIR prior features
= 395 features
```

---

## Final Paper-Aligned Configuration

> **Important:** the revised manuscript uses **SHAP Top320**, not the earlier Top200 setting.

| Component | Final setting |
|---|---|
| Drug representation | Mol2vec, 64 dimensions |
| Target representation | ProstT5, 1,024 dimensions before selection |
| ProstT5 training | Frozen pretrained encoder; no fine-tuning |
| Feature selection | Nested SHAP |
| Selected target features | Top320 |
| Pseudo-WkNNIR prior features | 11 dimensions |
| Drug-neighbor `k` | 10 |
| Target-neighbor `k` | 10 |
| Hash precision | 6 decimal places |
| Prior generation for training data | 3-fold inner OOF |
| PU-inspired reliability weighting | Enabled |
| Suspicious-negative threshold | 0.75 |
| Minimum negative-sample weight | 0.50 |
| PU down-weighting | Continuous |
| ENN | Enabled |
| ENN neighbors | 5 |
| ENN `kind_sel` | `mode` |
| Classifier | XGBoost + LightGBM |
| Fusion | Soft voting |
| XGBoost weight | 0.5 |
| LightGBM weight | 0.5 |
| Decision threshold | 0.5 |
| Outer evaluation | 10-fold stratified CV |
| Random state | 42 |
| Final independent test | One evaluation after configuration is fixed |

---

## Method

### 1. Pair-level data preprocessing

The original BindingDB train/validation/test files are first merged and cleaned globally rather than retaining the original split.

The preprocessing script:

- checks missing `SMILES`, `Target Sequence`, and `Label`;
- canonicalizes SMILES with RDKit;
- removes spaces/tabs/newlines from target sequences;
- converts target sequences to uppercase;
- replaces non-standard amino-acid symbols `U/Z/O/B` with `X`;
- restricts labels to `{0, 1}`;
- constructs a unique drug–target `pair_key`;
- removes fully duplicated drug–target–label triplets;
- removes all records belonging to drug–target pairs with conflicting labels;
- performs an 8:2 label-stratified re-split using `random_state = 42`.

### 2. Drug representation with Mol2vec

Each drug is encoded as a **64-dimensional Mol2vec vector**.

Final extraction settings include:

- RDKit SMILES parsing;
- `mol2alt_sentence`;
- molecular substructure radius = `1`;
- unseen token = `UNK`;
- strict invalid-SMILES checking;
- output columns `mol2vec_000` to `mol2vec_063`.

### 3. Target representation with ProstT5

ProstT5 is used as a **frozen pretrained encoder**. It is **not retrained or fine-tuned** in the revised workflow.

Target preprocessing and inference settings:

- input: FASTA sequences;
- uppercase conversion;
- `U/Z/O/B -> X`;
- spaces inserted between adjacent amino acids;
- `T5Tokenizer`;
- `T5EncoderModel`;
- evaluation mode;
- `torch.no_grad()`;
- maximum sequence length = `512`;
- batch size = `4`;
- pooling = first-token representation from the last hidden state;
- output columns `feature_0` to `feature_1023`.

The resulting drug and target embeddings form an initial **1,088-dimensional representation**:

```text
64 Mol2vec + 1,024 ProstT5 = 1,088 dimensions
```

### 4. Nested SHAP Top320 feature selection

SHAP selection is applied **only to the 1,024 ProstT5 target dimensions**. All 64 Mol2vec drug features are retained.

Within each outer training fold:

1. a SHAP-specific XGBoost model is fitted using only that training fold;
2. `TreeExplainer` is used to calculate feature contributions;
3. ProstT5 features are ranked by mean absolute SHAP value;
4. the Top320 target dimensions are retained;
5. the validation fold is transformed using the indices learned from the training fold only.

For the final independent-test experiment, SHAP selection is performed again using the complete development training set only.

### 5. Pseudo-WkNNIR local-prior features

The revised pipeline constructs **11 Pseudo-WkNNIR prior features** from local drug/target similarities.

Because explicit reusable drug IDs and target IDs are not required by the feature files, pseudo entity IDs are generated by fixed-precision hashing of drug and target representation vectors.

The prior module uses:

- cosine nearest neighbors;
- drug-neighbor `k = 10`;
- target-neighbor `k = 10`;
- six-decimal hashing precision;
- drug-neighborhood priors;
- target-neighborhood priors;
- joint drug–target-neighborhood priors;
- 3-fold inner out-of-fold prior construction for training samples.

The training sample's prior features are therefore not constructed directly from its own label.

### 6. PU-inspired reliability weighting

`prior_max_score` is used as a proxy for uncertainty associated with negative labels.

Negative samples with strong local positive evidence are **not relabeled** and are **not automatically deleted**. Instead, their contribution to classifier fitting is continuously down-weighted.

Default implementation:

```text
suspicious-negative threshold = 0.75
minimum negative weight       = 0.50
continuous down-weighting     = enabled
```

### 7. ENN cleaning

Edited Nearest Neighbours is applied **only to the training data**.

Final setting:

```text
n_neighbors = 5
kind_sel    = "mode"
```

When ENN retains a sample, its PU-derived sample weight is retained and passed to both XGBoost and LightGBM.

### 8. XGBoost–LightGBM soft-voting ensemble

Both classifiers are fitted on the same 395-dimensional representation.

The final predicted probability is:

```text
P = 0.5 * P_XGBoost + 0.5 * P_LightGBM
```

The final binary prediction uses:

```text
threshold = 0.5
```

---

## Dataset and Pair-Level Cleaning

The manuscript uses the curated BindingDB benchmark data distributed as three original subsets.

### Original data

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| Train | 12,667 | 6,334 | 6,333 |
| Validation | 6,642 | 927 | 5,715 |
| Test | 13,284 | 1,905 | 11,379 |
| **Total** | **32,593** | **9,166** | **23,427** |

### Global cleaning

- Original merged records: **32,593**
- Fully duplicated records collapsed: **4,538**
- Conflicting drug–target pairs: **805**
- Records removed because of conflicting labels: **3,318**
- Final unique and label-consistent records: **24,737**

### Final stratified 8:2 split

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| Development training | 19,789 | 4,626 | 15,163 |
| Independent test | 4,948 | 1,157 | 3,791 |

The independent test set is held out from feature selection, parameter selection, sample weighting, ENN cleaning, and model fitting.

---

## Performance

### Independent test performance

The revised final ProstEDTI model achieved:

| Metric | Value |
|---|---:|
| SN | **0.7882** |
| SP | **0.9301** |
| ACC | **0.8969** |
| MCC | **0.7141** |
| AUC | **0.9420** |
| AUPR | **0.8425** |

### Comparison with reproduced modern DTI baselines

All models in this comparison were retrained using the cleaned dataset and evaluated on the same independent test set.

| Model | SN | SP | ACC | MCC | AUC | AUPR |
|---|---:|---:|---:|---:|---:|---:|
| GraphDTA-GCN | 0.6992 | 0.9240 | 0.8715 | 0.6351 | 0.9151 | 0.7833 |
| GraphDTA-GIN | 0.7459 | 0.9206 | 0.8797 | 0.6651 | 0.9196 | 0.7961 |
| DGCNN | 0.6802 | 0.9248 | 0.8676 | 0.6216 | 0.9109 | 0.7799 |
| MolTrans | 0.6353 | **0.9465** | 0.8737 | 0.6282 | 0.9141 | 0.7894 |
| DTiTransformer | 0.6681 | 0.9172 | 0.8589 | 0.5983 | 0.9054 | 0.7548 |
| BarlowDTI | 0.7450 | 0.9203 | 0.8793 | 0.6640 | 0.9231 | 0.8019 |
| **ProstEDTI** | **0.7882** | 0.9301 | **0.8969** | **0.7141** | **0.9420** | **0.8425** |

### Paper-selected SHAP Top320 setting under 10-fold CV

The Top320 setting produced:

| Metric | Mean ± SD |
|---|---:|
| SN | 0.7707 ± 0.0179 |
| SP | 0.9275 ± 0.0083 |
| ACC | **0.8908 ± 0.0087** |
| MCC | **0.6963 ± 0.0233** |
| AUC | 0.9389 ± 0.0064 |
| AUPR | 0.8345 ± 0.0207 |

Top320 was selected because it provided the strongest overall ACC/MCC trade-off while reducing the original 1,024-dimensional target representation.

---

## Repository Structure

The repository contains both the existing prediction/application code and the revised paper-aligned reproducibility pipeline.

```text
ProstEDTI/
├── Predictor/                                  # Existing web/prediction application
├── Train/                                      # Earlier training utilities retained in the repository
│
├── BindingDB/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── final_drug_smi/                             # Cleaned/split drug SMILES files
├── final_target_fasta/                         # Cleaned/split target FASTA files
├── mol2vec/
│   ├── model.pkl                               # Pretrained Mol2vec model
│   └── ...
│
├── data_preprocessing_clean_pair_split.py      # Pair-level cleaning + stratified 8:2 split
├── integrated_mol2vec_ProstT5_feature_pipeline.py
│                                                  # Mol2vec + ProstT5 extraction/integration
├── final_train_cv_shap320_thr05_xgb_lgbm.py   # Final nested 10-fold CV pipeline
├── final_full_train_independent_test_shap320_thr05_xgb_lgbm.py
│                                                  # Full-train + independent-test pipeline
│
├── mol2vec_ProstT5_train.csv                   # Precomputed development features (Git LFS)
├── mol2vec_ProstT5_test.csv                    # Precomputed independent-test features (Git LFS)
│
├── final_shap320_cv_training/
│   ├── cv_fold_metrics.csv
│   ├── cv_summary_metrics.csv
│   ├── cv_summary_metrics_wide.csv
│   ├── cv_oof_predictions.csv
│   ├── cv_selected_features_by_fold.csv
│   ├── cv_sampling_distribution.csv
│   ├── cv_prior_meta.csv
│   ├── cv_pu_reliability_summary.csv
│   ├── model_manifest.csv
│   ├── models/
│   │   └── fold_XX_model_bundle.pkl            # Git LFS
│   └── shap_explainability/
│
├── final_shap320_full_train_independent_test/
│   ├── independent_test_metrics.csv
│   ├── independent_test_predictions.csv
│   ├── training_set_apparent_metrics_diagnostic_only.csv
│   ├── model/
│   │   └── final_full_train_model_bundle.pkl   # Git LFS
│   ├── process_files/
│   └── shap_explainability/
│
├── .gitattributes                              # Git LFS rules
└── README.md
```

### Which scripts reproduce the revised manuscript?

For the **revised paper results**, use the four root-level scripts:

```text
data_preprocessing_clean_pair_split.py
integrated_mol2vec_ProstT5_feature_pipeline.py
final_train_cv_shap320_thr05_xgb_lgbm.py
final_full_train_independent_test_shap320_thr05_xgb_lgbm.py
```

These implement the paper-aligned **Top320 + Pseudo-WkNNIR + PU + ENN + XGBoost/LightGBM** workflow.

---

## Installation

### 1. Clone the repository

Because the repository contains large CSV/model files managed with **Git LFS**, install Git LFS before retrieving the complete data/model artifacts.

```bash
git lfs install
git clone https://github.com/YHJ3298/ProstEDTI.git
cd ProstEDTI
git lfs pull
```

### 2. Python dependencies

Core packages used by the revised pipeline include:

```text
numpy
pandas
tqdm
joblib
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
matplotlib
rdkit
gensim
mol2vec
torch
transformers
sentencepiece
```

A typical installation is:

```bash
pip install numpy pandas tqdm joblib scikit-learn imbalanced-learn \
            xgboost lightgbm shap matplotlib rdkit gensim mol2vec \
            torch transformers sentencepiece
```

Depending on the platform and CUDA version, PyTorch may need to be installed separately using the appropriate official build.

### Environment used in the manuscript

| Component | Configuration |
|---|---|
| OS | Linux 5.15.0-78-generic, x86_64 |
| CPU | Intel Xeon Platinum 8470Q |
| RAM | 754 GiB |
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| NVIDIA driver | 580.105.08 |
| CUDA | CUDA 13.0 reported by `nvidia-smi`; toolkit 12.4.131 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.12.1 |
| RDKit | 2025.09.6 |
| mol2vec | 0.2.2 |
| gensim | 4.4.0 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.3.0 |
| LightGBM | 4.6.0 |
| imbalanced-learn | 0.14.2 |
| SHAP | 0.52.0 |
| NumPy | 2.4.0 |
| pandas | 3.0.3 |
| tqdm | 4.66.2 |

Exact reproduction is easiest with versions close to the environment above.

---

## Pretrained ProstT5 Model

The ProstT5 model is large and is **not included in the root training pipeline through ordinary Git files**.

Download the pretrained model from the official Rostlab repository:

https://huggingface.co/Rostlab/ProstT5

Place the downloaded model files under:

```text
./ProstT5/
```

so that the default feature-extraction command can load:

```text
--prostt5-model ./ProstT5
```

The revised ProstEDTI pipeline uses ProstT5 as a frozen encoder and does not fine-tune it.

---

## Quick Reproduction

If the precomputed feature matrices were successfully downloaded through Git LFS, the expensive Mol2vec/ProstT5 extraction stage can be skipped.

### 1. Reproduce nested 10-fold CV

```bash
python final_train_cv_shap320_thr05_xgb_lgbm.py \
    --train-input ./mol2vec_ProstT5_train.csv \
    --output-root ./final_shap320_cv_training \
    --top-protein-k 320 \
    --enn-n-neighbors 5 \
    --threshold 0.5 \
    --n-splits 10 \
    --random-state 42 \
    --xgb-weight 0.5 \
    --lgbm-weight 0.5
```

### 2. Train the final model and evaluate the independent test set

```bash
python final_full_train_independent_test_shap320_thr05_xgb_lgbm.py \
    --train-input ./mol2vec_ProstT5_train.csv \
    --test-input ./mol2vec_ProstT5_test.csv \
    --output-root ./final_shap320_full_train_independent_test \
    --top-protein-k 320 \
    --enn-n-neighbors 5 \
    --threshold 0.5 \
    --random-state 42 \
    --xgb-weight 0.5 \
    --lgbm-weight 0.5
```

The independent-test script:

1. fits `StandardScaler` on the complete development training set;
2. performs SHAP Top320 selection using the training set only;
3. generates 3-fold OOF priors for the training samples;
4. fits the full Pseudo-WkNNIR prior model and transforms the independent test set;
5. computes PU-inspired training weights;
6. applies ENN to the training data only;
7. trains the final XGBoost and LightGBM models;
8. predicts the independent test set once;
9. saves the final model bundle, predictions, metrics, and process metadata.

---

## Full Reproduction from Raw BindingDB Files

### Step 1. Pair-level cleaning and re-splitting

Expected input:

```text
./BindingDB/train.csv
./BindingDB/val.csv
./BindingDB/test.csv
```

Run:

```bash
python data_preprocessing_clean_pair_split.py
```

Default outputs include:

```text
./final_drug_smi/
./final_target_fasta/
./preprocessing_report/
```

### Step 2. Extract Mol2vec and ProstT5 features

Make sure the following models exist:

```text
./mol2vec/model.pkl
./ProstT5/
```

Run:

```bash
python integrated_mol2vec_ProstT5_feature_pipeline.py \
    --mol2vec-model ./mol2vec/model.pkl \
    --prostt5-model ./ProstT5 \
    --feature-output-dir ./feature_cache \
    --final-output-dir . \
    --target-batch-size 4 \
    --target-max-length 512 \
    --target-pooling first
```

This generates:

```text
./mol2vec_ProstT5_train.csv
./mol2vec_ProstT5_test.csv
```

with the label as the final column.

### Step 3. Run nested 10-fold CV

```bash
python final_train_cv_shap320_thr05_xgb_lgbm.py
```

### Step 4. Retrain on the complete development set and evaluate the independent test set

```bash
python final_full_train_independent_test_shap320_thr05_xgb_lgbm.py
```

---

## Output Files

### Preprocessing

`data_preprocessing_clean_pair_split.py` can generate:

```text
preprocessing_report/
├── basic_clean_pair_table_before_duplicate_handling.csv
├── clean_unique_pair_table.csv
├── final_train_pair_table.csv
├── final_test_pair_table.csv
├── removed_records.csv
├── preprocessing_summary.json
└── preprocessing_summary.csv
```

### Cross-validation

`final_shap320_cv_training/` contains:

- fold-level SN/SP/ACC/MCC/AUC/AUPR;
- mean and standard deviation across folds;
- out-of-fold predictions;
- selected features per fold;
- ENN sampling distributions;
- Pseudo-WkNNIR metadata;
- PU reliability summaries;
- fold model bundles;
- SHAP summary plots.

Important files:

```text
cv_fold_metrics.csv
cv_summary_metrics.csv
cv_summary_metrics_wide.csv
cv_oof_predictions.csv
model_manifest.csv
```

### Independent test

`final_shap320_full_train_independent_test/` contains:

```text
independent_test_metrics.csv
independent_test_predictions.csv
training_set_apparent_metrics_diagnostic_only.csv
model/final_full_train_model_bundle.pkl
process_files/
shap_explainability/
```

`training_set_apparent_metrics_diagnostic_only.csv` is provided only as a diagnostic and should **not** be interpreted as generalization performance.

---

## Implementation Details

### XGBoost defaults

The final scripts use the following main XGBoost settings:

```text
n_estimators      = 600
max_depth         = 7
learning_rate     = 0.05
subsample         = 0.9
colsample_bytree  = 0.9
min_child_weight  = 1
gamma             = 0
reg_alpha         = 0
reg_lambda        = 1
objective         = binary:logistic
eval_metric       = logloss
tree_method       = hist
```

### LightGBM defaults

```text
n_estimators       = 600
learning_rate      = 0.05
num_leaves         = 127
max_depth          = -1
min_child_samples  = 20
subsample          = 0.9
subsample_freq     = 1
colsample_bytree   = 0.9
reg_alpha          = 0
reg_lambda         = 1
objective          = binary
metric             = binary_logloss
```

### SHAP ranking model

The SHAP feature-ranking stage uses a separate XGBoost model:

```text
n_estimators      = 300
max_depth         = 5
learning_rate     = 0.05
subsample         = 0.8
colsample_bytree  = 0.8
tree_method       = hist
```

The ranking model is used for target-feature selection; it is distinct from the final soft-voting classifiers.

---

## Reproducibility and Leakage Control

The revised pipeline explicitly restricts all data-dependent operations to training data.

Within each outer CV fold:

```text
outer training fold
    ├── StandardScaler fit
    ├── SHAP ranking + Top320 selection
    ├── 3-fold inner OOF Pseudo-WkNNIR priors
    ├── PU-inspired sample weighting
    ├── ENN cleaning
    └── XGBoost + LightGBM fitting

outer validation fold
    └── evaluation only
```

The outer validation fold does **not** participate in:

- scaler fitting;
- SHAP ranking;
- feature selection;
- prior-model fitting;
- PU-weight estimation;
- ENN filtering;
- classifier fitting.

After all settings are fixed, the complete development training set is used to rebuild the final pipeline. The independent test set is then transformed and evaluated once.

This separation is important for avoiding optimistic performance estimates caused by feature-selection or preprocessing leakage.

---

## Interpretability and Case Study

The revised manuscript evaluates ProstEDTI interpretability at multiple levels:

- **SHAP** for global feature contribution;
- **LIME** for local sample-level explanations;
- **Mol2vec substructure tracing** for drug-side chemical interpretation;
- **K-mer auxiliary analysis** for target-side amino-acid interpretation.

The manuscript also reports a **SARS-CoV-2 3CL protease** case study using 15 antiviral compounds that were not present in the ProstEDTI training data.

Compounds with established or development-stage evidence related to 3CL protease inhibition, including **Ibuzatrelvir, Nirmatrelvir, Simnotrelvir, and MK-7845**, were predicted as positive in the case study.

These outputs should be interpreted as **computational screening and hypothesis-generation results**, not as experimental confirmation of binding or clinical efficacy.

---

## Important Notes

### 1. Revised manuscript vs. earlier repository settings

Older repository material may contain settings such as **SHAP Top200** or earlier performance numbers. Those settings do **not** represent the final revised manuscript.

For paper-aligned reproduction, use:

```text
SHAP Top320
Pseudo-WkNNIR 11D
PU reliability weighting
ENN k = 5
threshold = 0.5
XGBoost weight = 0.5
LightGBM weight = 0.5
final feature dimension = 395
```

### 2. Git LFS

Large feature matrices and model bundles are managed with Git LFS.

After cloning, run:

```bash
git lfs pull
```

If a `.csv` or `.pkl` file contains only a short LFS pointer instead of the actual data/model, Git LFS objects have not been downloaded correctly.

### 3. ProstT5 truncation

The final feature-extraction script uses:

```text
max_length = 512
```

Longer sequences are therefore truncated by the tokenizer in the final implementation.

### 4. Invalid SMILES

The default preprocessing/feature-extraction workflow uses strict validity checks. Invalid SMILES cause termination rather than silent skipping to avoid drug/target row misalignment.

### 5. Model scope

ProstEDTI is a statistical learning framework. It does not explicitly model all three-dimensional geometric constraints of a binding pocket, conformational dynamics, pharmacokinetics, toxicity, or clinical efficacy.

Predicted interactions should therefore be validated using appropriate downstream methods such as molecular docking, molecular dynamics simulations, biochemical assays, or cell-based experiments.

---

## Citation

The manuscript associated with this repository is:

> **Yun Zuo, Hongjin Yan, Xubin Wu, Fei Ge, Sirui Fei, Jingwen Liang, Qiao Ning.**  
> *ProstEDTI: Achieving high-precision prediction of drug-target interactions using the advanced pre-trained architecture ProstT5 and the ENN undersampling algorithm.*

Formal journal citation information and DOI can be added here after publication.

---

## Contact

**Corresponding author**  
Qiao Ning  
School of Artificial Intelligence and Computer Science, Jiangnan University  
Email: `ningq669@jiangnan.edu.cn`

**Project/code contact**  
Hongjin Yan  
Email: `1033220228@jiangnan.edu.cn`

For reproducibility questions, model-loading issues, or bug reports, please open a GitHub Issue:

https://github.com/YHJ3298/ProstEDTI/issues

---

## License

A standalone license file should be added to the repository to define reuse and redistribution terms explicitly. Until then, please do not infer a software license solely from this README.
