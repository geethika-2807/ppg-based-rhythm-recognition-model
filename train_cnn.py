"""
Step 5: Patient-aware train/val/test split + small 1D CNN for
regular/irregular classification on IBI windows.

Run locally:
    pip install scikit-learn tensorflow
    python train_cnn.py
"""

import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = 'dataset/ibi_windows_dataset.npz'
RANDOM_SEED = 42

# ---------- Load ----------
data = np.load(DATA_PATH, allow_pickle=True)
X, y, groups = data['X'], data['y'], data['record_ids']
print(f"Loaded X={X.shape}, y={y.shape}, {len(set(groups))} unique patients/records")

# ---------- Patient-aware split: 70% train, 15% val, 15% test ----------
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
train_idx, temp_idx = next(gss1.split(X, y, groups))

X_temp, y_temp, groups_temp = X[temp_idx], y[temp_idx], groups[temp_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X_temp[val_idx], y_temp[val_idx]
X_test, y_test = X_temp[test_idx], y_temp[test_idx]

print(f"\nTrain: {len(X_train)} windows, {len(set(groups[train_idx]))} patients, "
      f"{y_train.mean()*100:.1f}% irregular")
print(f"Val:   {len(X_val)} windows, {len(set(groups_temp[val_idx]))} patients, "
      f"{y_val.mean()*100:.1f}% irregular")
print(f"Test:  {len(X_test)} windows, {len(set(groups_temp[test_idx]))} patients, "
      f"{y_test.mean()*100:.1f}% irregular")

# ---------- Normalize (z-score using TRAIN stats only) ----------
mean, std = X_train.mean(), X_train.std()
X_train_n = (X_train - mean) / std
X_val_n = (X_val - mean) / std
X_test_n = (X_test - mean) / std

X_train_n = X_train_n[..., np.newaxis]  # (n, 10, 1)
X_val_n = X_val_n[..., np.newaxis]
X_test_n = X_test_n[..., np.newaxis]

# ---------- Model ----------
model = keras.Sequential([
    layers.Input(shape=(X_train_n.shape[1], 1)),
    layers.Conv1D(16, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv1D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
]

history = model.fit(
    X_train_n, y_train,
    validation_data=(X_val_n, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
)

# ---------- Evaluate ----------
test_loss, test_acc, test_auc = model.evaluate(X_test_n, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test AUC:      {test_auc:.4f}")

y_pred = (model.predict(X_test_n, verbose=0) > 0.5).astype(int).flatten()
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['regular', 'irregular']))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------- Save ----------
model.save('ppg_regularity_cnn.keras')
np.savez('normalization_stats.npz', mean=mean, std=std)
print("\nSaved model to ppg_regularity_cnn.keras")
print("Saved normalization stats to normalization_stats.npz (needed by the GUI later)")