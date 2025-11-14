import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

# Load data (do not use the first column as index, keep it as a feature column)
data = pd.read_csv('enn_train_ProstT5_shap.csv')

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels (now including the original first column as a feature)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
best_auc = 0  # For recording the best AUC
best_model_lgb = None  # For recording the best LightGBM model
best_model_xgb = None  # For recording the best XGBoost model
accuracies = []
auc_list = []
aupr_list = []
mcc_list = []

tp_list = []
fp_list = []
fn_list = []
tn_list = []

i = 0

for train_index, test_index in kf.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train LightGBM
    model_lgb = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_lgb.fit(X_train, y_train)
    y_prob_lgb = model_lgb.predict_proba(X_test)[:, 1]

    # Train XGBoost
    model_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_xgb.fit(X_train, y_train)
    y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

    # Ensemble prediction (average probabilities)
    y_prob_ensemble = 0.5 * y_prob_lgb + 0.5 * y_prob_xgb
    y_pred = (y_prob_ensemble >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob_ensemble)
    precision, recall, _ = precision_recall_curve(y_test, y_prob_ensemble)
    aupr_score = auc(recall, precision)
    mcc = matthews_corrcoef(y_test, y_pred)

    accuracies.append(accuracy)
    auc_list.append(auc_score)
    aupr_list.append(aupr_score)
    mcc_list.append(mcc)
    tp_list.append(((y_pred == 1) & (y_test == 1)).sum())
    fp_list.append(((y_pred == 1) & (y_test == 0)).sum())
    fn_list.append(((y_pred == 0) & (y_test == 1)).sum())
    tn_list.append(((y_pred == 0) & (y_test == 0)).sum())

    i += 1
    print(f"Fold {i}: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}, MCC = {mcc:.4f}")

    # Update best models if current AUC is better
    if auc_score > best_auc:
        best_auc = auc_score
        best_model_lgb = model_lgb
        best_model_xgb = model_xgb

# Save the best models
joblib.dump(best_model_lgb, "best_model_lgb.pkl")
joblib.dump(best_model_xgb, "best_model_xgb.pkl")

joblib.dump(scaler, "scaler.pkl")

# Calculate average metrics
mean_tp = np.mean(tp_list)
mean_fp = np.mean(fp_list)
mean_fn = np.mean(fn_list)
mean_tn = np.mean(tn_list)

sensitivity = mean_tp / (mean_tp + mean_fn)
specificity = mean_tn / (mean_tn + mean_fp)
mean_mcc = np.mean(mcc_list)
mean_acc = np.mean(accuracies)
mean_auc = np.mean(auc_list)
mean_aupr = np.mean(aupr_list)

print("\n===== 10-Fold Cross-Validation Results for LightGBM + XGBoost Ensemble =====")
print("Sensitivity (SN):", sensitivity)
print("Specificity (SP):", specificity)
print("Accuracy (ACC):", mean_acc)
print("Matthews Correlation Coefficient (MCC):", mean_mcc)
print("Average AUC:", mean_auc)
print("Average AUPR:", mean_aupr)