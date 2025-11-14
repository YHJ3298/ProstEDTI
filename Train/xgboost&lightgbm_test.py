import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

# Load data
train_data = pd.read_csv('enn_train_ProstT5_shap.csv', index_col=0)  # Training set
test_data = pd.read_csv('enn_test_ProstT5_shap.csv', index_col=0)    # Test set

# Shuffle training data (optional)
train_data = train_data.sample(frac=1).reset_index(drop=True)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values  # Training set features
y_train = train_data.iloc[:, -1].values   # Training set labels

X_test = test_data.iloc[:, :-1].values  # Test set features
y_test = test_data.iloc[:, -1].values   # Test set labels

# Normalize features
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)  # Use normalization parameters from training set

# Define LightGBM model
model_lgb = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='binary_logloss',
    learning_rate=0.01,
    n_estimators=500,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train LightGBM model
model_lgb.fit(X_train_normalized, y_train)
y_prob_lgb = model_lgb.predict_proba(X_test_normalized)[:, 1]  # Get probability of positive class

# Define XGBoost model
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.01,
    n_estimators=500,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train XGBoost model
model_xgb.fit(X_train_normalized, y_train)
y_prob_xgb = model_xgb.predict_proba(X_test_normalized)[:, 1]  # Get probability of positive class

# Ensemble prediction (weighted average probabilities)
y_prob_ensemble = 0.5 * y_prob_lgb + 0.5 * y_prob_xgb
y_pred = (y_prob_ensemble >= 0.5).astype(int)  # Classify based on ensemble probabilities

# Calculate performance metrics
tp = ((y_pred == 1) & (y_test == 1)).sum()
fp = ((y_pred == 1) & (y_test == 0)).sum()
fn = ((y_pred == 0) & (y_test == 1)).sum()
tn = ((y_pred == 0) & (y_test == 0)).sum()

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob_ensemble)
precision, recall, _ = precision_recall_curve(y_test, y_prob_ensemble)
aupr_score = auc(recall, precision)
mcc = matthews_corrcoef(y_test, y_pred)

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print results
print("\n===== Performance of LightGBM + XGBoost Ensemble Model on Independent Test Set =====")
print("Sensitivity (SN):", sensitivity)
print("Specificity (SP):", specificity)
print("Accuracy (ACC):", accuracy)
print("Matthews Correlation Coefficient (MCC):", mcc)
print("AUC:", auc_score)
print("AUPR:", aupr_score)