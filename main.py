import os
os.environ["KERAS_BACKEND"] = "torch"  # Keras'ın motorunu PyTorch yapıyoruz!

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# create app
app = Flask(__name__)

# load the trained model
model = load_model('model.h5')

# class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Eğer uploads klasörü yoksa otomatik oluşturuyoruz
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        
        if file and file.filename != '':
            # save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)
            
            # predict results
            result, confidence = predict_tumor(file_location)

            # HATA DÜZELTİLDİ: confidence_score yerine 'confidence' gönderildi.
            # HATA DÜZELTİLDİ: Resim yolu Flask static yapısına uygun hale getirildi.
            return render_template(
                'index.html', 
                result=result, 
                confidence=f"{confidence * 100:.2f}", 
                file_path=f"uploads/{file.filename}"
            )

    return render_template('index.html', result=None)

# python main - HATA DÜZELTİLDİ: '=' yerine '==' yapıldı kanka
if __name__ == '__main__':
    app.run(debug=True)