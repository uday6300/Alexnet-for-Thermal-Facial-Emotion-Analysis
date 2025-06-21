# Alexnet
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# Config
data_folder = "/Users/uday_iitm/Downloads/data 3/thermal"
img_size = (128, 128)
valid_emotions = ['anger', 'happy', 'fear', 'disgust', 'sadness', 'surprise']
label_map = {e: i for i, e in enumerate(valid_emotions)}

# Load function
def load_data_from_filenames(folder, img_size):
    X, y = [], []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        parts = fname.lower().split('_')
        found = [e for e in valid_emotions if e in parts]
        if not found:
            continue
        label = found[0]
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        y.append(label_map[label])
    return np.array(X), np.array(y)

X, y = load_data_from_filenames(data_folder, img_size)
print("Loaded:", X.shape, "images")

# Normalize and encode
X = X / 255.0
y_cat = to_categorical(y, num_classes=len(valid_emotions))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.4, stratify=y_cat, random_state=42)
alexnet = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(len(valid_emotions), activation='softmax')
])
alexnet.compile(optimizer=Adam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = alexnet.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      epochs=13,
                      batch_size=32)
# Accuracy
final_accuracy = alexnet.evaluate(x_test, y_test, verbose=0)[1]
print(f"\n🎯 Final Accuracy (AlexNet): {final_accuracy * 100:.2f}%")

# Predictions
y_pred = np.argmax(alexnet.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix with all labels
cm = confusion_matrix(y_true, y_pred)
annot = np.array([[str(cell) for cell in row] for row in cm])

plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=valid_emotions,
            yticklabels=valid_emotions,
            linewidths=1.0,
            linecolor='gray',
            square=True,
            annot_kws={"size": 14, "color": "black"})

plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title("Confusion Matrix (AlexNet - 6 Emotion Classes)", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Classification Report
print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=valid_emotions))
