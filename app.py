from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from keras.models import load_model
from numpy import  asarray, expand_dims, argmax
from base64 import b64decode

app = Flask(__name__)

print(tf.__version__)
print(tf.keras.__version__)

model = load_model('assets/multi_fashion_apparel_image_classifier', compile=True)
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
  image = tf.image.decode_image(decoded_image, channels=1)
  image = tf.image.resize(image, [28,28], method=tf.image.ResizeMethod.BILINEAR)
  image = asarray(image)
  image = image / 255.0
  image = expand_dims(image, axis=0)

  return image

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  request_body = request.get_json(force = True)
  image_data = request_body['image']
  decoded_image = b64decode(image_data)
  image = prepare_image(decoded_image)
  prediction = model.predict(image)
  highest_prediction = argmax(prediction)

  print('predicted: ', labels[highest_prediction])
  return jsonify({ 'prediction': labels[highest_prediction] })
