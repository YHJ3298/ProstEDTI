# ProstEDTI: A High-Precision Drug-Target Interaction Prediction System

## Project Overview
ProstEDTI is an open-source computational tool designed for high-accuracy and interpretable prediction of Drug-Target Interactions (DTI). Built on advanced pre-trained models and intelligent data processing strategies, the system integrates **Mol2vec molecular embedding technology** (capturing drug chemical structures) and **ProstT5 protein sequence-structure joint modeling technology** (fusing target sequence and spatial structure information) for dual-representation fusion. It also incorporates ENN (Edited Nearest Neighbours) undersampling, SHAP feature selection, and an XGBoost-LightGBM ensemble model to effectively address core challenges in current DTI prediction—insufficient feature representation, class imbalance, and poor model interpretability—providing reliable support for drug discovery and repurposing research.

On the BindingDB benchmark dataset, the system achieves state-of-the-art performance:
- Accuracy (ACC): 86.03%
- Matthews Correlation Coefficient (MCC): 0.7207
- Area Under ROC Curve (AUC): 0.9287
- Area Under Precision-Recall Curve (AUPR): 0.9204

The project provides complete predictor code, pre-trained classifiers, and data processing scripts to ensure research reproducibility and academic extensibility.


## Core Features
1. **Dual-Representation Fusion Strategy**: Integrates 64-dimensional Mol2vec molecular embeddings (preserving drug structural and physicochemical features) with 1024-dimensional ProstT5 protein embeddings (fusing sequence-structure information) to construct a comprehensive feature space covering all dimensions of drug-target interactions.
2. **Data & Feature Optimization**:
   - **ENN Undersampling**: Eliminates noisy samples through neighborhood consistency judgment to address class imbalance, improving the model’s ability to identify minority classes (effective drug-target pairs).
   - **SHAP Feature Selection**: Filters 200 key features from high-dimensional protein embeddings to reduce redundant information interference and enhance model training efficiency and generalization.
3. **High-Performance Ensemble Model**: Adopts a soft voting fusion strategy of XGBoost and LightGBM, leveraging the strengths of both algorithms in nonlinear feature learning and large-scale data processing to balance prediction accuracy and computational efficiency.
4. **Multi-Level Interpretability**: Combines SHAP-based global feature importance analysis and LIME-based local decision logic visualization to reveal the biological basis of model predictions, facilitating research on drug action mechanisms.
5. **Lightweight Prediction Workflow**: Prediction only requires code in the "Predictor" folder—no need to run pre-training scripts—lowering the barrier to use. Pre-training scripts are only for reference when reconstructing or improving the model.


## Directory Structure & File Description
```
ProstEDTI/
├── Predictor/                # Core prediction module (end-users only need this folder)
│   ├── models/              
│   │   ├── mol2vec/        
│   │   ├── ProstT5/         # Placeholder: User downloads ProstT5 model here manually
│   │   ├── best_model_lgb.pkl   
│   │   ├── best_model_xgb.pkl   
│   │   └── scaler.pkl          
│   ├── static/              
│   ├── templates/           
│   │   ├── index.html       
│   │   ├── upload.html      
│   │   ├── result.html      
│   │   └── contact.html    
│   ├── uploads/             
│   ├── app.py               # Flask entry (launches web service)
│   └── requirements.txt    
├── Train/  # For researchers (model reconstruction/improvement)
│   ├── 1_data preprocessing.py  
│   ├── 2_drug feature extraction_mol2vec.py 
│   ├── 3_target feature extraction ProstT5.py 
│   ├── 4_feature integration.py 
│   ├── 5_ENN_new.py             # ENN undersampling (balance training data)
│   ├── 6_shap_analysis.py       # SHAP feature selection (top 200 from 1024D)
│   ├── xgboost&lightgbm_10folds.py  
│   └── xgboost&lightgbm_test.py     
├── BindingDB/               # Original benchmark dataset
├── final_drug_smi/          
├── final_target_fasta/      
├── mol2vec_ProstT5_train_enn.csv  
├── mol2vec_ProstT5_test.csv   
├── enn_train_ProstT5_shap.csv # Final training features (post-SHAP, 264D)
├── enn_test_ProstT5_shap.csv  # Final test features (post-SHAP, 264D)
├── shap_feature_importance_bar.png  
├── shap_feature_summary_beeswarm.png 
└── README.md                
```

### Key File Usage Distinction
- **Prediction Only Requires the "Predictor" Folder**: Contains complete prediction functionality (web interface, batch prediction scripts, pre-trained classifiers). It can be launched directly—no need to interact with "Pre-training Related Scripts".
- **Pre-training Related Scripts**: Only for researchers to reference when reconstructing the model, adjusting parameters, or expanding datasets. Ordinary users do not need to run these scripts.


## Installation & Dependencies
### Prerequisites
- Python 3.8 or higher
- CUDA 11.0 or higher (GPU acceleration is recommended; CPU is supported but slower)
- Disk Space: At least 5GB (for storing predictor dependencies, Mol2vec model, and classifiers; does not include the ProstT5 model)

### Install Dependencies
Only dependencies in the "Predictor" folder need to be installed. Run the following commands:
```bash
# Navigate to the Predictor folder
cd Predictor
# Install dependencies
pip install -r requirements.txt
```


## Pre-Trained Model Preparation (Critical Notes)
### 1. Models That Do Not Require Manual Training (Provided)
The "Predictor/models/" directory already contains the following pre-trained files—no additional download is needed:
- `mol2vec/model.pkl`: Pre-trained Mol2vec molecular embedding model
- `best_model_lgb.pkl`, `best_model_xgb.pkl`: XGBoost/LightGBM ensemble classifiers
- `scaler.pkl`: Feature normalizer

### 2. ProstT5 Model Acquisition (Not Uploaded to GitHub; Download Manually)
Due to the large size of the ProstT5 model (>10GB), it is not uploaded to GitHub. Users must download it from the official Hugging Face repository. Follow these steps:
1. Visit the official ProstT5 Hugging Face repository: [https://huggingface.co/Rostlab/ProstT5](https://huggingface.co/Rostlab/ProstT5)
2. Download all complete model files (including `config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.)
3. Create a new `ProstT5` folder under "Predictor/models/" and place all downloaded model files directly in this folder
4. Verify the path: `Predictor/models/ProstT5/` (ensure model files are stored directly in this path with no additional subdirectories)


## Usage Guide
### 1. Data Preparation
Prepare an input file in CSV format with **two mandatory columns** (column names must match exactly, case-sensitive):
- `SMILES`: Drug molecular structure (e.g., "CCO" for ethanol)
- `Target Sequence`: Protein target sequence (composed of uppercase amino acid letters; no spaces or line breaks)

Example input file format:
| SMILES       | Target Sequence                          |
|--------------|------------------------------------------|
| CN(O)S=Na    | MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH |
| CC1(C)[C@H](C(O)=O)N2[C@@H](CC2)S1 | MSRSLLLRFLYLLKPISTGQFGDFKVYQTSALSGKRLGNNPVLCFCGPGFPEVYGTFMWDGYNGWSHWCQSCT |

### 2. Quick Start for Prediction (Web Interface, Recommended)
Only operate within the "Predictor" folder—no pre-training scripts need to be run:
```bash
# Navigate to the Predictor folder
cd Predictor
# Launch the Flask web service (default port: 5001)
python app.py
```
Once the service is running, access `http://localhost:5001` in your browser to perform the following operations:
- **Homepage**: View system introduction, core technical framework, and performance metrics.
- **Prediction**: Upload a CSV file → Click "Start Prediction" → Wait for results (≈10-20 seconds per sample, depending on sample size).
- **Result Download**: After prediction completes, click "Download Results (CSV)" to obtain the output file containing prediction probabilities and labels.

### 3. Batch Prediction (Command Line, for Developers)
For batch processing of large datasets, write a batch script in the "Predictor" folder (refer to the prediction logic in `app.py`). The core steps are:
1. Call `mol2vec` to extract drug features (using `Predictor/models/mol2vec/model.pkl`).
2. Call `ProstT5` to extract target features (using the model in `Predictor/models/ProstT5/`).
3. Load `scaler.pkl` to normalize features.
4. Load `best_model_xgb.pkl` and `best_model_lgb.pkl` for ensemble prediction.
5. Export prediction results to a CSV file.


## Critical Notes
1. **ProstT5 Model Path**: Ensure the ProstT5 model files are stored in `Predictor/models/ProstT5/`. If you modify the path, update the `app.config['PROSTT5_PATH']` configuration in `app.py` accordingly.
2. **Prediction Data Limit**: The web interface recommends uploading ≤50 samples at a time (to avoid GPU memory overflow). For batch prediction, adjust the `batch_size` (modify in the ProstT5 feature extraction function).
3. **File Encoding**: Input CSV files must use UTF-8 encoding to avoid garbled text or format errors that prevent reading.
4. **Pre-Training Script Note**: "Pre-training Related Scripts" are only for model reconstruction (e.g., adjusting feature dimensions, replacing datasets). Ordinary users do not need to run these scripts to avoid overwriting pre-trained classifiers.


## Contact & Support
- **Contact The Author**: Hongjin Yan (Email: 1033220228@jiangnan.edu.cn)
- **Research Team**: School of Artificial Intelligence and Computer Science, Jiangnan University, China
- **Code Repository**: [https://github.com/YHJ3298/ProstEDTI](https://github.com/YHJ3298/ProstEDTI) (replace with your actual GitHub repository URL)

For issues with ProstT5 model download, prediction function bugs, or feature requests, submit an Issue on GitHub or contact the corresponding author directly for technical support.


## License
This project is open-source under the MIT License. See the LICENSE file (to be added to the project root directory) for details. Commercial and non-commercial use is permitted, provided that the original author’s copyright information is retained.
