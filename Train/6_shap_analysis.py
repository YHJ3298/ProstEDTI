import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('mol2vec_ProstT5_train_enn.csv')
test_df = pd.read_csv('mol2vec_ProstT5_test.csv')

# Extract mol2vec feature columns (modified to mol2vec_000 to mol2vec_063)
mol2vec_cols = [f'mol2vec_{str(i).zfill(3)}' for i in range(0, 64)]

# Extract Protein feature columns (from feature_0 to feature_1023)
protein_cols = [f'feature_{i}' for i in range(0, 1024)]

X_train = train_df[protein_cols].values
y_train = train_df['label'].values
X_test = test_df[protein_cols].values
y_test = test_df['label'].values

# Train the model
model = XGBClassifier(random_state=42, objective='binary:logistic')
model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 1. Bar plot (average SHAP values)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=protein_cols, show=False)
plt.title("SHAP Feature Importance (Bar)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_feature_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Beeswarm plot (SHAP value distribution)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train, feature_names=protein_cols, show=False)
plt.title("SHAP Feature Summary (Beeswarm)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_feature_summary_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

# Calculate SHAP importance (mean of absolute values)
shap_abs = np.abs(shap_values).mean(axis=0)

# Construct feature importance DataFrame and select Top200
feature_importance = pd.DataFrame({
    'feature': protein_cols,
    'importance': shap_abs
}).sort_values('importance', ascending=False)

selected_200 = feature_importance.head(200)['feature'].tolist()

# Extract data (and rename columns)
selected_train_feat = train_df[selected_200].copy()
selected_test_feat = test_df[selected_200].copy()

# Rename to feature_1 to feature_200
new_feature_names = [f'feature_{i+1}' for i in range(200)]
selected_train_feat.columns = new_feature_names
selected_test_feat.columns = new_feature_names

# Check and filter out actually existing mol2vec columns
existing_mol2vec_cols_train = [col for col in mol2vec_cols if col in train_df.columns]
existing_mol2vec_cols_test = [col for col in mol2vec_cols if col in test_df.columns]

# Retain actually existing mol2vec columns
selected_train_final = pd.concat([train_df[existing_mol2vec_cols_train], selected_train_feat], axis=1)
selected_test_final = pd.concat([test_df[existing_mol2vec_cols_test], selected_test_feat], axis=1)

# Add labels
selected_train_final['label'] = y_train
selected_test_final['label'] = y_test

# Save files
selected_train_final.to_csv('enn_train_ProstT5_shap.csv', index=False)
selected_test_final.to_csv('enn_test_ProstT5_shap.csv', index=False)

print(f"Saved successfully! Number of selected features: {len(selected_200)}")
print("Training set file: enn_train_ProstT5_shap.csv")
print("Test set file: enn_test_ProstT5_shap.csv")