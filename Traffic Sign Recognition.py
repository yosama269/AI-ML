import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Paths
# =========================
dataset_folder = r'C:\Users\20111\gtsrb_dataset'
train_csv = os.path.join(dataset_folder, 'Train.csv')
test_csv = os.path.join(dataset_folder, 'Test.csv')

# =========================
# Load CSV files
# =========================
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

print("Train CSV columns:", train_data.columns)
print("Test CSV columns:", test_data.columns)

# =========================
# Function to load images
# =========================
def load_images(data, folder_path):
    images = []
    labels = []
    print("Loading images...")
    for _, row in data.iterrows():
        img_path = os.path.join(folder_path, row['Path'].replace('/', os.sep))
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32))
            images.append(img)
            labels.append(row['ClassId'])
        else:
            print(f"Warning: {img_path} not found")
    return np.array(images), np.array(labels)

# =========================
# Load images
# =========================
X_train, y_train = load_images(train_data, dataset_folder)
X_test, y_test = load_images(test_data, dataset_folder)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# =========================
# Normalize images
# =========================
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# =========================
# One-hot encode labels
# =========================
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# =========================
# Manual train/validation split
# =========================
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# =========================
# Data augmentation
# =========================
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train_split)

# =========================
# Build CNN model
# =========================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# Train model
# =========================
history = model.fit(
    datagen.flow(X_train_split, y_train_split, batch_size=32),
    epochs=30,
    validation_data=(X_val_split, y_val_split),
    shuffle=True
)

# =========================
# Evaluate model
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

# =========================
# Predictions and confusion matrix
# =========================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy from sklearn
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy (sklearn): {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
