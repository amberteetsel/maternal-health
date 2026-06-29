# Dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
svm_rec = os.path.join(BASE_DIR, "resources", "svm")

# ================================================================================
# DATA PREP / PREPROCESSING
# ================================================================================

# Load health data
health_data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health_preprocessed.csv"))

# Build Abortion Restriction Set
### states where only restriction is for "partial birth abortions"
zero_restrict = [
    'ME', 'VT', 'WA', 'MN', 'IL', 'NY', 'RI', 'MA', 'OR', 'NV', 'CT',
    'CA', 'CO', 'MD', 'DE', 'NH', 'PA', 'NJ', 'VA', 'NM'
]

# Add binary labels to health data
df = health_data.copy()
df['Abortion_Restricted'] = np.where(df['State'].isin(zero_restrict), 0, 1)

# Pare down features
features_all = [
    'Inadequate Prenatal Care',
    'No Cancer Screen',
    'Infant Mortality',
    # 'Low Birth Weight',
    'Maternal Mortality',
    'Maternity Care Desert',
    'Maternity Practices Score',
    # 'Postpartum Depression',
    'No Postpartum Visit',
    # 'Severe Maternal Morbidity',
    'Unintended Pregnancy',
    'No Preventative',
    'Patients Per Doctor',
    ]
df = df[['State', 'Year', 'Abortion_Restricted'] + features_all]
# Post-Dobbs filter
df = df.loc[df.Year >= 2022].reset_index(drop=True)
# Save Copy of Data for Site
df.to_csv(os.path.join(svm_rec, "svm_preprocessed.csv"), index=False)

# Statistics by Abortion_Restricted
## higher numbers = worse health outcomes
feature_means = df.drop(columns=['Year', 'State']).groupby('Abortion_Restricted').mean().T
feature_means['Delta'] = feature_means[1] - feature_means[0]
feature_means['Delta (%)'] = feature_means['Delta']/feature_means[0]
feature_means.rename(columns={
    0: 'No (0)',
    1: 'Yes (1)'
}, inplace=True)
feature_means.sort_values(by='Delta (%)', ascending=False, inplace=True)
feature_means.to_csv(os.path.join(svm_rec, "feature_means.csv"))

## Preparing Training and Testing Sets

# Split into Features/Target
X = df.copy().drop(columns=['State', 'Year', 'Abortion_Restricted'])
y = df.copy()['Abortion_Restricted']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training, Testing sets for display on website
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df.insert(0, 'Abortion_Restricted', y_train.values)
test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df.insert(0, 'Abortion_Restricted', y_test.values)
train_df.to_csv(os.path.join(svm_rec, "train_display.csv"), index=False)
test_df.to_csv(os.path.join(svm_rec, "test_display.csv"), index=False)


# ================================================================================
# MODELING & RESULTS
# ================================================================================

### SVM - Test multiple kernels, costs

# Hyperparameters
kernels = ['linear', 'poly', 'rbf']
costs = [0.1, 1, 10]

results = {}

# Iterate thru kernels and costs
for kernel in kernels:
    results[kernel] = {}

    for C in costs:

        # build model
        svm_model = SVC(
            kernel = kernel,
            C = C,
            random_state=42
        )

        # fit on data
        svm_model.fit(X_train_scaled, y_train)

        # make predictions
        y_pred = svm_model.predict(X_test_scaled)

        # evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # store results
        results[kernel][C] = {
            'accuracy': acc,
            'confusion_matrix': cm
        }
        print(f"Kernel: {kernel}, Cost: {C}  |  Accuracy = {acc:3f}")

# Results DataFrame
accuracy_dict = {
    kernel: {cost: data['accuracy'] for cost, data in costs.items()}
    for kernel, costs in results.items()
}
result_df = pd.DataFrame.from_dict(accuracy_dict, orient='index')
result_df.rename(columns={
    0.1: "Accuracy (Cost = 0.1)",
    1.0: "Accuracy (Cost = 1.0)",
    10.0: "Accuracy (Cost = 10.0)"
}, inplace=True)
result_df.index.name = 'Kernel'

# save for display on website
result_df.to_csv(os.path.join(svm_rec, "accuracy_table_svm.csv"))

# Viz - Confusion Matrixes for Best Result (Each Kernel)
kernel_choice = 'linear'
cost_choise = 1
best_cm = results[kernel_choice][cost_choise]['confusion_matrix']
plt.figure(figsize=(5,4))
sns.heatmap(best_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            linecolor='white',
            linewidths=1.5,
            xticklabels=['No Restriction', 'Restriction'], 
            yticklabels=['No Restriction', 'Restriction'])
plt.title(f"Confusion Matrix: {kernel_choice.capitalize()} Kernel (C={cost_choise:.1f})")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(svm_rec, f"best_{kernel_choice}_cm.png"))

# Viz - Confusion Matrixes for Best Result (Each Kernel)
kernel_choice = 'poly'
cost_choise = 1
best_cm = results[kernel_choice][cost_choise]['confusion_matrix']
plt.figure(figsize=(5,4))
sns.heatmap(best_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            linecolor='white',
            linewidths=1.5,
            xticklabels=['No Restriction', 'Restriction'], 
            yticklabels=['No Restriction', 'Restriction'])
plt.title(f"Confusion Matrix: {kernel_choice.capitalize()} Kernel (C={cost_choise:.1f})")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(svm_rec, f"best_{kernel_choice}_cm.png"))

# Viz - Confusion Matrixes for Best Result (Each Kernel)
kernel_choice = 'rbf'
cost_choise = 1
best_cm = results[kernel_choice][cost_choise]['confusion_matrix']
plt.figure(figsize=(5,4))
sns.heatmap(best_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            linecolor='white',
            linewidths=1.5, 
            xticklabels=['No Restriction', 'Restriction'], 
            yticklabels=['No Restriction', 'Restriction'])
plt.title(f"Confusion Matrix: {kernel_choice.upper()} Kernel (C={cost_choise:.1f})")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(svm_rec, f"best_{kernel_choice}_cm.png"))

# Reshape accuracy data for plotting
df_acc = pd.read_csv(os.path.join(svm_rec, 'accuracy_table_svm.csv'))
df_melt = df_acc.melt(id_vars='Kernel', var_name='Cost', value_name='Accuracy')
df_melt['Cost'] = df_melt['Cost'].str.extract(r'Cost = ([\d\.]+)').astype(float)

# Plotting performance curves
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df_melt, x='Cost', y='Accuracy', hue='Kernel', marker='o', linewidth=2.5, ax=ax,
             palette='flare', legend=False)
ax.set_xscale('log')
ax.set_xticks([0.1, 1.0, 10.0])
ax.set_xticklabels(['0.1', '1.0', '10.0'])
ax.set_title('SVM Model Accuracy by Kernel and Cost Parameter (C)', loc='left', fontweight='bold', pad=15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_ylabel('Testing\nAccuracy', rotation=0, va='center', ha='right', labelpad=10)
ax.set_xlabel('Cost Parameter (C) - Log Scale', labelpad=10)
ax.text(s="RBF Kernel", x=1.05, y=0.97, color=sns.color_palette('flare')[-1], fontweight='bold')
ax.text(s="Linear Kernel", x=0.1, y=0.87, color=sns.color_palette('flare')[1], fontweight='bold')
ax.text(s="Poly Kernel", x=0.5, y=0.745, color=sns.color_palette('flare')[3], fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(svm_rec, 'svm_accuracy_comparison.png'))

