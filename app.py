from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import keras
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

print(tf.__version__)
print(keras.__version__)

model = keras.models.load_model('assets/multi_fashion_apparel_image_classifier', compile=True)
labels = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
]

def prepare_image(decoded_image):
  image = Image.open(io.BytesIO(decoded_image)).convert('L').resize((28,28))
  image = np.asarray(image)
  image = image / 255.0
  image = np.expand_dims(image, axis=0)

  return image

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  request_body = request.get_json(force = True)
  image_data = request_body['image']
  decoded_image = base64.b64decode(image_data)
  image = prepare_image(decoded_image)
  prediction = model.predict(image)
  highest_prediction = np.argmax(prediction)

  print('predicted: ', labels[highest_prediction])
  return jsonify({ 'prediction': labels[highest_prediction] })
