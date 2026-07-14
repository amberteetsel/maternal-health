# import first to fix mac bug
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nn_rec = os.path.join(BASE_DIR, "resources", "neural_net")

# Load Data
data = pd.read_csv(os.path.join(nn_rec, "health_x_policy.csv"))
# print(data.shape)
# print(f"Missing Values: \n{data.isna().sum()}")

# Clean Column Names
data.columns = data.columns.str.replace(' ', '_')
data.columns = data.columns.str.lower()

# Isolate Health Metrics
health_cols = [col for col in data.columns if col not in ['state', 'year', 'abortion_restricted']]

# Clean copy
df = data.copy()

# U.S. Census Bureau Regional Mapping
region_map = {
    # NORTHEAST
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast', 
    'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast', 'RI': 'Northeast', 
    'VT': 'Northeast',
    
    # MIDWEST
    'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 
    'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest', 
    'ND': 'Midwest', 'OH': 'Midwest', 'SD': 'Midwest', 'WI': 'Midwest',
    
    # SOUTH
    'AL': 'South', 'AR': 'South', 'DE': 'South', 'FL': 'South', 
    'GA': 'South', 'KY': 'South', 'LA': 'South', 'MD': 'South', 
    'MS': 'South', 'NC': 'South', 'OK': 'South', 'SC': 'South', 
    'TN': 'South', 'TX': 'South', 'VA': 'South', 'WV': 'South', 
    'DC': 'South',
    
    # WEST
    'AK': 'West', 'AZ': 'West', 'CA': 'West', 'CO': 'West', 
    'HI': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 
    'NM': 'West', 'OR': 'West', 'UT': 'West', 'WA': 'West', 
    'WY': 'West'
}

df['region'] = df.state.map(region_map)

# Save clean copy for website display
tmp = df.copy()
tmp.columns = tmp.columns.str.replace('_', ' ')
tmp.columns = tmp.columns.str.title()
tmp.to_csv(os.path.join(nn_rec, 'nn_preprocessed.csv'), index=False)

# Features
X_features = df[health_cols].copy()

# dummy variables for region
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
X = pd.concat([X_features, region_dummies], axis=1)

# target var
y = df['abortion_restricted'].values.astype('float32')

# Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Snapshots of training/testing for report
X_train_display = X_train.copy()
X_train_display.insert(0, 'abortion_restricted', y_train)
X_train_display.insert(0, 'row_index', X_train_display.index.values)

X_test_display = X_test.copy()
X_test_display.insert(0, 'abortion_restricted', y_test)
X_test_display.insert(0, 'row_index', X_test_display.index.values)

X_train_display.to_csv(os.path.join(nn_rec, 'nn_train_display.csv'), index=False)
X_test_display.to_csv(os.path.join(nn_rec, 'nn_test_display.csv'), index=False)

# Scale health metrics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configure Binary Classification NN Model
tf.random.set_seed(42)

nn_model = models.Sequential([

    layers.Input(shape=(X_train_scaled.shape[1],)),

    # hidden layer
    layers.Dense(8, activation='relu'),

    # output layer (single node w Sigmoid activation)
    layers.Dense(1, activation='sigmoid')
])

# binary crossentropy loss
nn_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

# early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience=15,
    restore_best_weights=True
)

# Training
print('Starting model training...')
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split = 0.15,
    epochs = 60,
    callbacks=[early_stop],
    batch_size = 16,
    verbose=1
)
print('Success! Model training complete.')

# Test Evaluation & Confusion Matrix
test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)

# predictions
y_pred_probs = nn_model.predict(X_test_scaled)
## convert probs into binary classification
y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_classes)

# plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Protected (0)', 'Restricted (1)'], 
            yticklabels=['Protected (0)', 'Restricted (1)'])
plt.xlabel('Predicted', labelpad=10)
plt.ylabel('Actual', labelpad=10)
plt.title(f'Confusion Matrix (Test Accuracy: {test_acc:.2%})', pad=15, fontweight='bold')
plt.savefig(os.path.join(nn_rec, 'nn_confusion_matrix.png'), bbox_inches='tight', dpi=300)
# plt.show()

# Classification Report for Display
rpt = classification_report(y_test, y_pred_classes, output_dict=True)
cdf = pd.DataFrame(rpt)
cdf = cdf.rename(columns={"0.0": 'Target Class: Protected (0)',
                          "1.0": 'Target Class: Restricted (1)',
                          'macro avg': 'Macro Average'})
cdf = cdf.iloc[:3, [0,1,3]]
cdf.index = cdf.index.str.capitalize()
cdf = round(cdf,4)
cdf = cdf.reset_index()
cdf = cdf.rename(columns={'index': 'Evaluation Metric'})
cdf.to_csv(os.path.join(nn_rec, "classification_report_nn.csv"), index=False)