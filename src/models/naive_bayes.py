# Dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

# Base Directory
notebook_dir = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(notebook_dir, "../.."))

# resources
nb_rec = os.path.join(BASE_DIR, "resources", "nbayes")
if not os.path.exists(nb_rec):
    os.makedirs(nb_rec)

# Load Data (Births)
data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth", "birth_icu_processed.csv"))
feature_cols = ['maternal_age_group', 'maternal_race', 'maternal_ed', 'binned_visits', 'binned_cigs',
                'pay_source', 'delivery_method',
                'diabetes', 'hypertension', 'eclampsia', 'induction', 'augmentation', 'steroids', 'antibiotics', 'anesthesia',
                'maternal_transfusion', 'perineal_lac', 'ruptured_uterus', 'unplanned_hysterectomy']
target_col = ["icu_admit"]
data['binned_visits'] = data['binned_visits'].fillna("UNKNOWN")

# Organize features/target
X_cat = data[feature_cols]
y_mask = data['icu_admit'].isin(['Y', 'N'])     # drop Unknowns
X = X_cat[y_mask]
y = data['icu_admit'][y_mask].map({'Y': 1, 'N': 0}).astype(int)

# Encode
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Features matrix shape: {X_train.shape}")
print(f"Training target class balance:\n{y_train.value_counts()}")

# Save samples of training/testing set for website
X_train_export = pd.DataFrame(columns=feature_cols, data=X_train)
y_train_export = pd.DataFrame(columns=target_col, data=y_train)
X_test_export = pd.DataFrame(columns=feature_cols, data=X_test)

X_train_export.to_csv(os.path.join(nb_rec, "X_train_nb.csv"), index=False)
X_test_export.to_csv(os.path.join(nb_rec, "X_test_nb.csv"), index=False)
y_train_export.to_csv(os.path.join(nb_rec, "y_train_nb.csv"), index=False)

# MODELING

# alpha=1.0 for Laplace smoothign
nb_model = CategoricalNB(alpha = 1.0)
nb_model.fit(X_train, y_train)

# predictions on test set
y_pred = nb_model.predict(X_test)
y_proba = nb_model.predict_proba(X_test)[:, 1]

# print results
print("="*60)
print("             NAIVE BAYES CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['No ICU (0)', 'ICU Admit (1)']))

print("\n" + "="*60)
print("                      CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Correctly predicted No ICU):  {cm[0][0]}")
print(f"False Positives (Predicted ICU, was No ICU):   {cm[0][1]}")
print(f"False Negatives (Predicted No ICU, was ICU):   {cm[1][0]}")
print(f"True Positives (Correctly predicted ICU):     {cm[1][1]}")

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred).T
n_samples = cm.sum()

cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No ICU (0)", "ICU Admit (1)"])

fig, ax = plt.subplots(figsize=(6,6))
cm_disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='')

# update data labels
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = (count / n_samples) * 100
        new_lab = f"{count:,}\n({pct:.1f}%)"
        ax.images[0].axes.texts[i * cm.shape[i] + j].set_text(new_lab)

plt.title("Confusion Matrix", fontweight='bold', pad=15)
plt.xlabel("Prediction", labelpad=10)
plt.ylabel("Actual", ha='right', rotation=0, labelpad=10)
plt.savefig(os.path.join(nb_rec, "nb_cm.png"))

# Save and export classification report
report_dict = classification_report(
    y_test,
    y_pred,
    target_names=["No ICU (0)", "ICU Admit (1)"],
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(nb_rec, "nb_report.csv"))

# Build Performance Metrics Table
metric = [
    'Overall Accuracy',
    'ROC AUC Score',
    'ICU Class Precision',
    'ICU Class Recall',
    'ICU Class F-1 Score'
    ]

metric_vals = [
    report_df.iloc[2,0],
    roc_auc_score(y_test, y_proba),
    report_df.iloc[1,0],
    report_df.iloc[1,1],
    report_df.iloc[1,2]
    ]

formatted_vals = [
    f"{metric_vals[0]:.0%}",
    f"{metric_vals[1]:.4f}",
    f"{metric_vals[2]:.0%}",
    f"{metric_vals[3]:.0%}",
    f"{metric_vals[4]:.0%}",
]

interpretation = [
    "Model correctly classifies 89% of cases overall, though this metric is heavily driven by the majority class ('No ICU')",
    "Strong overall discriminative ability; the model is capable of separating high-risk mothers from low-risk mothers",
    "When the model predicts a mother will require the ICU, it is correct 74% of the time",
    "The model successfully identifies only 50% of the mothers who actually require the ICU, missing the other 50%",
    "The harmonic mean of precision and recall highlights a severe operational imbalance"
]

results_df = pd.DataFrame(columns=['Metric', 'Score/Value', 'Interpretation'])
results_df['Metric'] = metric
results_df['Score/Value'] = formatted_vals
results_df['Interpretation'] = interpretation
results_df.to_csv(os.path.join(nb_rec, "nb_results_interpret.csv"), index=False)
