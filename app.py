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

model = keras.models.load_model('assets/multi_fashion_apparel_image_classifier.h5')
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

def prepare_image(image):
  image = image.resize((28, 28))
  image = keras.preprocessing.image.img_to_array(image)
  greyscale_image = image[:, :, 0]
  image = np.expand_dims(greyscale_image, axis=0)

  return image

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  request_body = request.get_json(force = True)
  image_data = request_body['image']
  decoded_image = base64.b64decode(image_data)
  image = Image.open(io.BytesIO(decoded_image))
  prepared_image = prepare_image(image)
  prediction = model.predict(prepared_image)
  highest_prediction = np.argmax(prediction)

  print(prediction)
  print('predicted: ', labels[highest_prediction])
  return jsonify({ 'prediction': labels[highest_prediction] })