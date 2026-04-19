import tensorflow as tf
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Import Mobilenet
from libs.mobilenet.Codes.MobileNet_1DCNN import MobileNet 

# Data Preparation
df = pd.read_csv("data.csv")

# Clean strings and filter
df['Category'] = df['Category'].astype(str).str.lower().str.strip()
df = df[df['Category'].isin(['paper', 'stone'])].copy()
df['Category'] = df['Category'].map({'paper': 0, 'stone': 1})

# Append features
feature_cols = []
for i in range(21):
    feature_cols.extend([f"{i}_x", f"{i}_y"])

X = df[feature_cols].values
y = df['Category'].values

# Shuffle and Scale Feature
X, y = shuffle(X, y, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for 1D CNN (Samples, Features, Channels)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 42, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.15, random_state=42, stratify=y
)

# Init MobileNetV3
mobile_factory = MobileNet(
    length=42, 
    num_channel=1, 
    num_filters=32, 
    problem_type='Classification', 
    output_nums=2
)
model = mobile_factory.MobileNet_v3_Large()

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=12, 
    restore_best_weights=True
)

print(f"Training...")
model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=128,
    validation_data=(X_test, y_test), 
    callbacks=[early_stop]
) 

# Save Models
if not os.path.exists('Models'): os.makedirs('Models')

# Save full model
model.save('Models/hand_gesture_model.keras')
# Save scaler
joblib.dump(scaler, 'Models/scaler.pkl')

print("Model and Scaler Saved.")