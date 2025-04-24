import os
import sys
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Ensure UTF-8 encoding for the terminal output
sys.stdout.reconfigure(encoding='utf-8')

# Function to load images and labels
def load_data(data_path):
    categories = ['healthy', 'tumor']
    data = []
    labels = []
    for label, category in enumerate(categories):
        folder = os.path.join(data_path, category)
        if not os.path.exists(folder):
            print(f"Warning: Folder '{folder}' does not exist.")
            continue
        for file in os.listdir(folder):
            try:
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))  # Resize image to 64x64
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
                continue
    return np.array(data), to_categorical(labels)

# Set data paths for training and testing
train_data_path = r"C:\Users\anmol\Desktop\brain_tumor_detection-using_CNN\brain_tumor_dataset\Training"
test_data_path = r"C:\Users\anmol\Desktop\brain_tumor_detection-using_CNN\brain_tumor_dataset\Validation"

# Load and preprocess the data
x_train, y_train = load_data(train_data_path)
x_test, y_test = load_data(test_data_path)

# Normalize pixel values to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two classes: 'healthy' and 'tumor'
])

# Compile the model
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the trained model
model.save("model.h5")
print("✅ Model saved as model.h5")
