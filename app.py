from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import sqlite3

app = Flask(__name__)

# Load the trained model
model = load_model('emotion_model.h5')
# In app.py
model = load_model('emotion_model.h5')
print("Model loaded successfully!")  # ADD THIS LINE
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(img):
    # Preprocess image: grayscale, resize to 48x48, normalize
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel
    img = np.expand_dims(img, axis=0)  # Add batch
    pred = model.predict(img)
    return emotions[np.argmax(pred)]

# Initialize database
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (name TEXT, image BLOB, emotion TEXT)''')
conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Anonymous')
    image_data = request.form['image'].split(',')[1]  # Base64 data URL
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))
    emotion = predict_emotion(img)
    
    # Log to database
    c.execute("INSERT INTO users (name, image, emotion) VALUES (?, ?, ?)", (name, image_bytes, emotion))
    conn.commit()
    
    return jsonify({'emotion': emotion})

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name', 'Anonymous')
    file = request.files['file']
    file.stream.seek(0)  # Reset stream
    image_bytes = file.stream.read()
    img = Image.open(io.BytesIO(image_bytes))
    emotion = predict_emotion(img)
    
    # Log to database
    c.execute("INSERT INTO users (name, image, emotion) VALUES (?, ?, ?)", (name, image_bytes, emotion))
    conn.commit()
    
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    # In app.py, change last line to:
    app.run(debug=True, port=5001)