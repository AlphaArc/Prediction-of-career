from operator import mod
import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging
import pandas as pd 
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import json

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
cors=CORS(app)

data = None
model = None
target_classnames = None

@app.before_first_request
def startup():
  global data, model, target_classnames
  df = pd.read_csv('CWPP.csv')
  target = df.target
  df.drop('target_classnames', axis=1, inplace=True)
  df.drop('target', axis=1, inplace=True)
  data = df.to_numpy()
  target_classnames = ['Data Scientist','Full Stack Developer','Big Data Engineer','Database Administrator','Cloud Architect','Cloud Services Developer ','Network Architect','Data Quality Manager' ,'Machine Learning','Business Analyst']
  model = MultinomialNB(alpha=1)
  model.fit(data, target)
  
  
@app.route("/", methods=['POST', 'GET'])
def predict():
  if request.method == 'POST':
    request_data = request.get_json()
    array = [request_data['data']]
    predict = model.predict(array)
    predict_levels = model.predict_proba(array)
    print(predict_levels[0])
    # The above code is for the levels which you need to order
    return {'prediction': target_classnames[predict[0]]}
  
if __name__=='__main__':
    app.run(debug=True)
    
