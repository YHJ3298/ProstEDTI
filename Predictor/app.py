import os
import sys
import tempfile
import shutil
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import PandasTools
from models.mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import numpy as np
import shap
from xgboost import XGBClassifier
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Replace with random key in production (e.g., os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['MODEL_FOLDER'] = 'models'
app.config['PROSTT5_PATH'] = 'models/ProstT5'  # Replace with your actual ProstT5 model path
app.config['MOL2VEC_MODEL'] = os.path.join(app.config['MODEL_FOLDER'], 'mol2vec', 'model.pkl')
app.config['TRAIN_DATA'] = 'mol2vec_ProstT5_train_enn.csv'  # Training data path

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Data preprocessing function
def preprocess_data(input_file):
    """Generate temporary smi drug file and fasta target file"""
    temp_drug_folder = os.path.join(tempfile.gettempdir(), "temp_drug_smi")
    temp_target_folder = os.path.join(tempfile.gettempdir(), "temp_target_fasta")
    os.makedirs(temp_drug_folder, exist_ok=True)
    os.makedirs(temp_target_folder, exist_ok=True)
    try:
        df = pd.read_csv(input_file)
        app.logger.info(f"Successfully read dataset: {input_file}")
    except FileNotFoundError:
        raise Exception(f"File not found: {input_file}")
    required_cols = ['SMILES', 'Target Sequence']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise Exception(f"Dataset missing required columns: {missing_cols}")
    df['sample_index'] = range(len(df))
    app.logger.info(f"Data preprocessing completed ({len(df)} samples total)")
    # Generate drug smi file
    smi_data = pd.DataFrame({
        'Smiles': df['SMILES'],
        'ID': [f'molecule_{i + 1}' for i in range(len(df))]
    })
    smi_path = os.path.join(temp_drug_folder, 'test_drug.smi')
    smi_data[['Smiles', 'ID']].to_csv(smi_path, sep='\t', index=False)
    # Generate target fasta file
    fasta_path = os.path.join(temp_target_folder, 'test_target.txt')
    with open(fasta_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f">P{row['sample_index']}\n{row['Target Sequence']}\n")
    return temp_drug_folder, temp_target_folder, df[['SMILES', 'Target Sequence']]

# mol2vec feature extraction function
def extract_mol2vec_features(temp_drug_folder):
    input_file = os.path.join(temp_drug_folder, "test_drug.smi")
    output_file = os.path.join(tempfile.gettempdir(), "test_drug.csv")
    file_ext = input_file.split('.')[-1].lower()
    if file_ext in ['smi', 'ism']:
        df = pd.read_csv(
            input_file,
            delimiter='\t',
            usecols=[0, 1],
            names=['Smiles', 'ID'],
            on_bad_lines='skip'
        )
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol='Smiles')
    else:
        raise ValueError("Unsupported drug file format")
    df = df[df['ROMol'].notnull()].reset_index(drop=True)
    df['mol_sentence'] = df['ROMol'].apply(
        lambda mol: mol2alt_sentence(mol, radius=1)
    )
    try:
        model = word2vec.Word2Vec.load(app.config['MOL2VEC_MODEL'])
    except Exception as e:
        raise Exception(f"Failed to load mol2vec model: {str(e)}")
    vectors = sentences2vec(
        df['mol_sentence'],
        model,
        unseen="UNK"
    )
    feat_cols = [f"mol2vec_{i:03d}" for i in range(vectors.shape[1])]
    feat_df = pd.DataFrame(vectors, columns=feat_cols)
    feat_df.to_csv(output_file, index=False)
    return output_file

# ProstT5 feature extraction function (GPU memory optimized)
def extract_prostt5_features(temp_target_folder):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_index = 0
        app.logger.info(f"Using GPU device: {device} (index: {gpu_index})")
    else:
        device = torch.device("cpu")
        gpu_index = None
        app.logger.info(f"No GPU detected, using CPU device: {device}")
    # GPU memory optimization
    if gpu_index is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(0.7, device=gpu_index)
            torch.cuda.empty_cache()
            app.logger.info("GPU memory limit set successfully")
        except Exception as e:
            app.logger.warning(f"Failed to set GPU memory limit (does not affect operation): {str(e)}")
    # Verify model path
    model_path = app.config['PROSTT5_PATH']
    if not os.path.exists(model_path):
        raise Exception(f"ProstT5 model path does not exist: {model_path}")
    app.logger.info(f"Loading ProstT5 model: {model_path}")
    # Define file paths
    input_file = os.path.join(temp_target_folder, "test_target.txt")
    output_file = os.path.join(tempfile.gettempdir(), "test_target.csv")
    try:
        # Load tokenizer and model (half precision)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5EncoderModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
        app.logger.info("ProstT5 model loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load ProstT5 model: {str(e)}")
    # Parse FASTA file
    data = []
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
        protein_name = ""
        sequence = ""
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                if protein_name:
                    data.append((protein_name, " ".join(sequence)))
                protein_name = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence:
            data.append((protein_name, " ".join(sequence)))
        app.logger.info(f"FASTA file parsing completed, {len(data)} sequences total")
    except Exception as e:
        raise Exception(f"Failed to parse FASTA file: {str(e)}")
    # Batch feature extraction
    try:
        sequences = [seq[1] for seq in data]
        inputs = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        # Build data loader (batch_size=1)
        dataset = TensorDataset(input_ids, attention_mask)
        data_loader = DataLoader(dataset, batch_size=1)
        # Extract features
        all_embeddings = []
        for batch in tqdm(data_loader, desc="Extracting ProstT5 features", unit="sample"):
            batch_input_ids, batch_attention_mask = batch
            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        app.logger.info(f"ProstT5 feature extraction completed, dimension: {all_embeddings.shape}")
    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")
    # Save features
    try:
        column_names = [f'feature_{i}' for i in range(all_embeddings.shape[1])]
        df = pd.DataFrame(all_embeddings, columns=column_names)
        df.to_csv(output_file, index=False)
        app.logger.info(f"ProstT5 features saved to: {output_file}")
        return output_file
    except Exception as e:
        raise Exception(f"Failed to save features: {str(e)}")

# Feature merging function
def merge_features(drug_file, target_file):
    drug_df = pd.read_csv(drug_file)
    target_df = pd.read_csv(target_file)
    if len(drug_df) != len(target_df):
        raise ValueError(f"Mismatched number of drug features ({len(drug_df)}) and target features ({len(target_df)})")
    merged_df = pd.concat([drug_df, target_df], axis=1)
    merged_file = os.path.join(tempfile.gettempdir(), 'mol2vec_ProstT5_test.csv')
    merged_df.to_csv(merged_file, index=False)
    return merged_file

# SHAP feature dimensionality reduction function
def shap_dim_reduction(merged_file):
    train_file = app.config['TRAIN_DATA']
    if not os.path.exists(train_file):
        raise Exception(f"Training data file does not exist: {train_file}")
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(merged_file)
    except Exception as e:
        raise Exception(f"Failed to read data: {str(e)}")
    # Define feature columns
    mol2vec_cols = [f'mol2vec_{str(i).zfill(3)}' for i in range(0, 64)]
    protein_cols = [f'feature_{i}' for i in range(0, 1024)]
    # Prepare training data
    X_train = train_df[protein_cols].values
    y_train = train_df['label'].values
    # Train XGB model
    model = XGBClassifier(random_state=42, objective='binary:logistic')
    model.fit(X_train, y_train)
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    # Select Top200 features
    shap_abs = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': protein_cols,
        'importance': shap_abs
    }).sort_values('importance', ascending=False)
    selected_200 = feature_importance.head(200)['feature'].tolist()
    # Process test set features
    selected_test_feat = test_df[selected_200].copy()
    new_feature_names = [f'feature_{i + 1}' for i in range(200)]
    selected_test_feat.columns = new_feature_names
    # Retain mol2vec features
    existing_mol2vec_cols_test = [col for col in mol2vec_cols if col in test_df.columns]
    # Concatenate features
    selected_test_final = pd.concat([test_df[existing_mol2vec_cols_test], selected_test_feat], axis=1)
    output_file = os.path.join(tempfile.gettempdir(), 'enn_test_ProstT5_shap.csv')
    selected_test_final.to_csv(output_file, index=False)
    return output_file

# Temporary file cleaning function
def clean_temp_files(temp_drug_folder, temp_target_folder):
    if os.path.exists(temp_drug_folder):
        shutil.rmtree(temp_drug_folder, ignore_errors=True)
    if os.path.exists(temp_target_folder):
        shutil.rmtree(temp_target_folder, ignore_errors=True)
    # Clean intermediate files
    intermediate_files = [
        os.path.join(tempfile.gettempdir(), 'test_drug.csv'),
        os.path.join(tempfile.gettempdir(), 'test_target.csv'),
        os.path.join(tempfile.gettempdir(), 'mol2vec_ProstT5_test.csv'),
        os.path.join(tempfile.gettempdir(), 'enn_test_ProstT5_shap.csv')
    ]
    for file in intermediate_files:
        if os.path.exists(file):
            os.remove(file)

# Prediction function (ensemble model)
def drug_target_prediction(test_data_path, original_data):
    try:
        test_data = pd.read_csv(test_data_path)
        app.logger.info(f"Loading test data: {test_data_path} ({len(test_data)} samples)")
    except Exception as e:
        raise Exception(f"Failed to load test data: {str(e)}")
    # Separate features and labels
    if 'label' in test_data.columns:
        X_test = test_data.drop('label', axis=1).values
        y_true = test_data['label'].values
        has_true_label = True
    else:
        X_test = test_data.values
        y_true = None
        has_true_label = False
    # Load model and scaler
    try:
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        lgb_model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model_lgb.pkl')
        model_lgb = joblib.load(lgb_model_path)
        xgb_model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model_xgb.pkl')
        model_xgb = joblib.load(xgb_model_path)
    except Exception as e:
        raise Exception(f"Failed to load model/scaler: {str(e)}")
    # Feature normalization
    try:
        X_test_normalized = scaler.transform(X_test)
    except Exception as e:
        raise Exception(f"Feature normalization failed: {str(e)}")
    # Ensemble prediction
    try:
        y_prob_lgb = model_lgb.predict_proba(X_test_normalized)[:, 1]
        y_prob_xgb = model_xgb.predict_proba(X_test_normalized)[:, 1]
        y_prob_ensemble = 0.5 * y_prob_lgb + 0.5 * y_prob_xgb
        y_pred = (y_prob_ensemble >= 0.5).astype(int)
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")
    # Generate results
    result_df = original_data.copy()
    result_df["index"] = test_data.index
    result_df["pred_prob"] = y_prob_ensemble
    result_df["pred_label"] = y_pred
    if has_true_label:
        result_df["true_label"] = y_true
    # Save results
    output_path = os.path.join(tempfile.gettempdir(), "prediction_results.csv")
    result_df.to_csv(output_path, index=False)
    return result_df, output_path

# ---------------------- Flask Routes ----------------------
@app.route('/')
def index():
    """Homepage: System introduction + technical framework"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction route: GET returns upload page, POST processes prediction"""
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('predict'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('predict'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            temp_drug_folder = None
            temp_target_folder = None
            original_data = None
            try:
                # 1. Data preprocessing
                temp_drug_folder, temp_target_folder, original_data = preprocess_data(upload_path)
                # 2. Feature extraction
                drug_feat_file = extract_mol2vec_features(temp_drug_folder)
                target_feat_file = extract_prostt5_features(temp_target_folder)
                # 3. Feature merging
                merged_file = merge_features(drug_feat_file, target_feat_file)
                # 4. SHAP dimensionality reduction
                reduced_file = shap_dim_reduction(merged_file)
                # 5. Model prediction
                result_df, result_path = drug_target_prediction(reduced_file, original_data)
                # Prepare result data
                result_data = result_df.to_dict('records')
                columns = result_df.columns.tolist()
                return render_template('result.html',
                                       result_data=result_data,
                                       columns=columns,
                                       filename=filename)
            except Exception as e:
                flash(f'Processing error: {str(e)}')
                app.logger.error(f"Processing error: {str(e)}")
                return redirect(url_for('predict'))
            finally:
                # Clean files
                if temp_drug_folder and temp_target_folder:
                    clean_temp_files(temp_drug_folder, temp_target_folder)
                if os.path.exists(upload_path):
                    os.remove(upload_path)
        else:
            flash('Only CSV files are supported')
            return redirect(url_for('predict'))

@app.route('/download')
def download_result():
    """Download prediction results"""
    result_path = os.path.join(tempfile.gettempdir(), "prediction_results.csv")
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True, download_name="ProstEDTI_prediction_results.csv")
    else:
        flash('No result file available for download')
        return redirect(url_for('predict'))

@app.route('/contact')
def contact():
    """Contact Us page"""
    return render_template('contact.html')

if __name__ == '__main__':
    # Matplotlib font configuration (for Chinese display if needed)
    import matplotlib
    matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    app.run(debug=True, port=5001)  # Set debug=False in production