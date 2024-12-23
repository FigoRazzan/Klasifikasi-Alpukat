from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Load model
model = load_model('model/cnn_model.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = []
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diunggah'
        
        files = request.files.getlist('file')
        
        for file in files:
            if file.filename == '':
                continue
            
            if file:
                # Simpan file
                image_path = os.path.join('static/uploads', file.filename)
                file.save(image_path)
                
                # Baca dan proses gambar
                image = cv2.imread(image_path)
                image = cv2.resize(image, (128, 128))  # Resize gambar
                image = np.expand_dims(image, axis=0) / 255.0  # Normalisasi
                
                # Prediksi
                pred = model.predict(image)
                label = np.argmax(pred, axis=1)[0]
                predictions.append({
                    'image': file.filename,
                    'label': "Matang" if label == 1 else "Mentah"
                })
    
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True) 