from flask import Flask,render_template,url_for,request,redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
# import pandas as pd
import sys
import os
import glob
import re
import pickle

# from __future__ import division, print_function

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten,Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization  



app = Flask(__name__)

model_path = "models\\RNN"
# Load your trained model
model = load_model(model_path)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28,28))
    img = np.array(img, dtype="float") / 255.0

    # Preprocessing the image
    # img = img.img_to_array(img)
    # x = np.true_divide(x, 255)
    img = np.expand_dims(img, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # img = preprocess_input(img, mode='caffe')

    pred = model.predict_classes(img)
    return (pred[0])

@app.route('/')
def home():

    return render_template('index1.html',title='Home')


@app.route('/predict',methods=['GET','POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        file_name = secure_filename(f.filename)
        


        # Save the file to ./uploads
        basepath =os.path.dirname(os.path.abspath(__file__))
        # basepath = 'C:\\Users\\fenil\\OneDrive\\Desktop\\E2E DL\\E2E_CNN_shoes_classification\\uploads'
        file_path = os.path.join(basepath,'static/uploads',file_name)
        f.save(file_path)

        # Make prediction
        vector = np.vectorize(model_predict)
        pred = vector(file_path,model)
        classes = ["Boots", "Sandals", "Slippers"]

        pred = np.array_str(pred)
        result = int(pred) 
        result = classes[result]           # Convert to string
        return render_template('layout01.html',result=result,file_name =file_name)
    return None

@app.route('/display/<file_name>')
def display_image(file_name):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static',filename='uploads/' + file_name), code=301)
port = int(os.environ.get('PORT',5000))
if __name__ == "__main__":
    app.run(debug=1,host='0.0.0.0',port=port) # or True   
# </file_name>

          
