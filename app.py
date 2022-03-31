from flask import Flask, request, render_template
import tensorflow as tf
import keras
import numpy as np
# from scipy.misc import imresize
# from tensorflow import keras

app = Flask(__name__)

print(tf.__version__)
print(keras.__version__)

model = keras.models.load_model('assets/multi_fashion_apparel_image_classifier.h5')


@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  img_data = request.get_data()
  # im = imresize(img_data, 28, 28)
  # im.reshape(1, 28, 28)
  return 'Yeah'