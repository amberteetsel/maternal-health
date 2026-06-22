import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# resources
tree_rec = os.path.join(BASE_DIR, "resources", "trees")
if not os.path.exists(tree_rec):
    os.makedirs(tree_rec)

# Load Data (Births)
data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth", "birth_icu_processed.csv"))
feature_cols = ['maternal_age_group', 'maternal_race', 'maternal_ed', 'binned_visits', 'binned_cigs',
                'pay_source', 'delivery_method',
                'diabetes', 'hypertension', 'eclampsia', 'induction', 'augmentation', 'steroids', 'antibiotics', 'anesthesia',
                'maternal_transfusion', 'perineal_lac', 'ruptured_uterus', 'unplanned_hysterectomy']
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

# ==============================================================================
# 1: SHALLOW (Max Depth = 3)
# Optimized for high readability and immediate clinical logic paths
# ==============================================================================
tree_shallow = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
tree_shallow.fit(X_train, y_train)
y_pred_shallow = tree_shallow.predict(X_test)

report_dict_shallow = classification_report(
    y_test,
    y_pred_shallow,
    target_names=["No ICU (0)", "ICU Admit (1)"],
    output_dict=True
)

report_df_shallow = pd.DataFrame(report_dict_shallow).transpose()
report_df_shallow.to_csv(os.path.join(tree_rec, "dt_report_shallow.csv"))

# ==============================================================================
# 2. MEDIUM (Max Depth = 5)
# Captures mid-level compound feature interactions without over-complicating
# ==============================================================================
tree_medium = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
tree_medium.fit(X_train, y_train)
y_pred_medium = tree_medium.predict(X_test)

report_dict_medium = classification_report(
    y_test,
    y_pred_medium,
    target_names=["No ICU (0)", "ICU Admit (1)"],
    output_dict=True
)

report_df_medium = pd.DataFrame(report_dict_medium).transpose()
report_df_medium.to_csv(os.path.join(tree_rec, "dt_report_medium.csv"))

# ==============================================================================
# 3: DEEP
# Maximizes recall by searching deep into overlapping combinations of complications
# ==============================================================================
tree_deep = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
tree_deep.fit(X_train, y_train)
y_pred_deep = tree_deep.predict(X_test)

report_dict_deep = classification_report(
    y_test,
    y_pred_deep,
    target_names=["No ICU (0)", "ICU Admit (1)"],
    output_dict=True
)

report_df_deep = pd.DataFrame(report_dict_deep).transpose()
report_df_deep.to_csv(os.path.join(tree_rec, "dt_report_deep.csv"))

# ==============================================================================
# PRINT ACTUAL METRICS
# ==============================================================================
for name, model in [("Shallow Tree", tree_shallow), ("Medium Tree", tree_medium), ("Deep Tree", tree_deep)]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n" + "="*40)
    print(f" ACTUAL RESULTS FOR: {name}")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, proba):.4f}")
    
    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# shallow
plt.figure(figsize=(20, 10))
plot_tree(tree_shallow, feature_names=feature_cols, class_names=['No ICU', 'ICU'], filled=True, rounded=True, fontsize=10)
plt.title("Tree Variation 1: Shallow Clinical Tree (Depth=3)", fontsize=16)
plt.savefig(os.path.join(tree_rec, "shallow_tree.png"), bbox_inches='tight', dpi=300)
# plt.close()
# plt.show()

# medium
plt.figure(figsize=(25, 12))
plot_tree(tree_medium, feature_names=feature_cols, class_names=['No ICU', 'ICU'], filled=True, rounded=True, fontsize=8)
plt.title("Tree Variation 2: Medium Risk Tree (Depth=5)", fontsize=16)
plt.savefig(os.path.join(tree_rec, "medium_tree.png"), bbox_inches='tight', dpi=300)
# plt.close()
# plt.show()

# Build Performance Comparison Table
metrics = [
    'Overall Accuracy',
    'ICU Class Recall',
    'ICU Class Precision',
    'ROC AUC Score',
    'True Negatives',
    'False Positives (False Alarms)',
    'False Negatives (Missed Patients)',
    'True Positives',
    'Total Structural Splits'
]

performance_cols_dt = ['Metric', 'Shallow Tree\n(Depth=3)', 'Medium Tree\n(Depth=5)', 'Deep Tree\n(Depth=10)']

tree1 = [
    f"{report_df_shallow.iloc[2,0]:.0%}",
    f"{report_df_shallow.iloc[1,1]:.0%}",
    f"{report_df_shallow.iloc[1,0]:.0%}",
    f"{roc_auc_score(y_test, y_pred_shallow):.4f}",
    f"{confusion_matrix(y_test, y_pred_shallow)[0,0]:,}",
    f"{confusion_matrix(y_test, y_pred_shallow)[0,1]:,}",
    f"{confusion_matrix(y_test, y_pred_shallow)[1,0]:,}",
    f"{confusion_matrix(y_test, y_pred_shallow)[1,1]:,}",
    f"{7:,}"
]
tree2 = [
    f"{report_df_medium.iloc[2,0]:.0%}",
    f"{report_df_medium.iloc[1,1]:.0%}",
    f"{report_df_medium.iloc[1,0]:.0%}",
    f"{roc_auc_score(y_test, y_pred_medium):.4f}",
    f"{confusion_matrix(y_test, y_pred_medium)[0,0]:,}",
    f"{confusion_matrix(y_test, y_pred_medium)[0,1]:,}",
    f"{confusion_matrix(y_test, y_pred_medium)[1,0]:,}",
    f"{confusion_matrix(y_test, y_pred_medium)[1,1]:,}",
    f"{31:,}"
]
tree3 = [
    f"{report_df_deep.iloc[2,0]:.0%}",
    f"{report_df_deep.iloc[1,1]:.0%}",
    f"{report_df_deep.iloc[1,0]:.0%}",
    f"{roc_auc_score(y_test, y_pred_deep):.4f}",
    f"{confusion_matrix(y_test, y_pred_deep)[0,0]:,}",
    f"{confusion_matrix(y_test, y_pred_deep)[0,1]:,}",
    f"{confusion_matrix(y_test, y_pred_deep)[1,0]:,}",
    f"{confusion_matrix(y_test, y_pred_deep)[1,1]:,}",
    f"{1023:,}"
]


performance_df_dt = pd.DataFrame({
    'Metric': metrics,
    'Shallow Tree (Depth=3)': tree1,
    'Medium Tree (Depth=5)': tree2,
    'Deep Tree (Depth=10)': tree3
})
performance_df_dt.to_csv(os.path.join(tree_rec, "dt_results.csv"), index=False)

# ==============================================================================
# CONFUSION MATRICES
# ==============================================================================

# Confusion Matrix - Shallow
cm_shallow = confusion_matrix(y_test, y_pred_shallow).T
n_samples = cm_shallow.sum()

cm_shallow_disp = ConfusionMatrixDisplay(confusion_matrix=cm_shallow, display_labels=["No ICU (0)", "ICU Admit (1)"])

fig, ax = plt.subplots(figsize=(6,6))
cm_shallow_disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='')

# update data labels
for i in range(cm_shallow.shape[0]):
    for j in range(cm_shallow.shape[1]):
        count = cm_shallow[i, j]
        pct = (count / n_samples) * 100
        new_lab = f"{count:,}\n({pct:.1f}%)"
        ax.images[0].axes.texts[i * cm_shallow.shape[i] + j].set_text(new_lab)

plt.title("Confusion Matrix (Shallow Tree)", fontweight='bold', pad=15)
plt.xlabel("Prediction", labelpad=10)
plt.ylabel("Actual", ha='right', rotation=0, labelpad=10)
plt.savefig(os.path.join(tree_rec, "dt_cm_shallow.png"), dpi=300)

# Confusion Matrix - Medium
cm_medium = confusion_matrix(y_test, y_pred_medium).T
n_samples = cm_medium.sum()

cm_medium_disp = ConfusionMatrixDisplay(confusion_matrix=cm_medium, display_labels=["No ICU (0)", "ICU Admit (1)"])

fig, ax = plt.subplots(figsize=(6,6))
cm_medium_disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='')

# update data labels
for i in range(cm_medium.shape[0]):
    for j in range(cm_medium.shape[1]):
        count = cm_medium[i, j]
        pct = (count / n_samples) * 100
        new_lab = f"{count:,}\n({pct:.1f}%)"
        ax.images[0].axes.texts[i * cm_medium.shape[i] + j].set_text(new_lab)

plt.title("Confusion Matrix (Medium Tree)", fontweight='bold', pad=15)
plt.xlabel("Prediction", labelpad=10)
plt.ylabel("Actual", ha='right', rotation=0, labelpad=10)
plt.savefig(os.path.join(tree_rec, "dt_cm_medium.png"), dpi=300)

# Confusion Matrix - Deep
cm_deep = confusion_matrix(y_test, y_pred_deep).T
n_samples = cm_deep.sum()

cm_deep_disp = ConfusionMatrixDisplay(confusion_matrix=cm_deep, display_labels=["No ICU (0)", "ICU Admit (1)"])

fig, ax = plt.subplots(figsize=(6,6))
cm_deep_disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='')

# update data labels
for i in range(cm_deep.shape[0]):
    for j in range(cm_deep.shape[1]):
        count = cm_deep[i, j]
        pct = (count / n_samples) * 100
        new_lab = f"{count:,}\n({pct:.1f}%)"
        ax.images[0].axes.texts[i * cm_deep.shape[i] + j].set_text(new_lab)

plt.title("Confusion Matrix (Deep Tree)", fontweight='bold', pad=15)
plt.xlabel("Prediction", labelpad=10)
plt.ylabel("Actual", ha='right', rotation=0, labelpad=10)
plt.savefig(os.path.join(tree_rec, "dt_cm_deep.png"), dpi=300)