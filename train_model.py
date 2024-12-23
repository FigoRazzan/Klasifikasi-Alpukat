import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_dataset(base_dir):
    data = []
    labels = []
    
    folder_mentah = os.path.join(base_dir, 'mentah')
    print(f"Isi folder mentah: {os.listdir(folder_mentah)}")  # Debugging
    for img_file in os.listdir(folder_mentah):
        print(f"Memproses file: {img_file}")  # Debugging
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_mentah, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                data.append(image)
                labels.append(0)
                print(f"Memproses {img_path} - Kelas: mentah")

    folder_matang = os.path.join(base_dir, 'matang')
    print(f"Isi folder matang: {os.listdir(folder_matang)}")  # Debugging
    for img_file in os.listdir(folder_matang):
        print(f"Memproses file: {img_file}")  # Debugging
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_matang, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                data.append(image)
                labels.append(1)
                print(f"Memproses {img_path} - Kelas: matang")

    return np.array(data), np.array(labels)

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 kelas: matang dan mentah
    return model

def train_and_save_model():
    # Load dataset
    print("Memulai loading dataset...")
    dataset_path = 'D:\Download\Documents\TUGAS\ITENAS\TUGAS\Semester 5\AI\Tubes\Klasifikasi Alpukat\dataset'  # Sesuaikan jalur ini
    data, labels = load_dataset(dataset_path)
    
    if len(data) == 0:
        print("Tidak ada data yang berhasil dimuat!")
        return
    
    # Normalisasi data
    data = data.astype('float32') / 255.0
    labels = to_categorical(labels, num_classes=2)  # One-hot encoding

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Buat model CNN
    model = create_cnn_model(input_shape=(128, 128, 3))
    
    # Kompilasi model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Latih model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Simpan model
    model.save('model/cnn_model.h5')
    print("Model berhasil disimpan di model/cnn_model.h5")

if __name__ == "__main__":
    print(tf.__version__)
    train_and_save_model() 