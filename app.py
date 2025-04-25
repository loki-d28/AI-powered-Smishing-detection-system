from flask import Flask, request, render_template,send_file ,redirect, url_for
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
app=Flask(__name__)
# Load model and vectorizer
model = load_model('smish_model.h5')  # Load the model
with open('tfidf_vec.pkl', 'rb') as f:
    tfidf_vec = pickle.load(f)

# Preprocessing input text function
def preprocess_input(text):
    result = re.sub('[^a-zA-Z]', ' ', text)
    result = result.lower()
    result = result.split()
    result = ' '.join([word for word in result if word not in stopwords.words('english')])
    return result
@app.route('/')
def home():
    return send_file('web.html')
@app.route('/about')
def about():
    return send_file('about.html')
@app.route('/support')
def support():
    return send_file('support.html')
@app.route('/C')
def guard():
    return send_file('C.html')
@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        user_msg = request.form.get('message')
#user_msg="Hi, your monthly subscription for Netflix has been renewed successfully."
        preprocess = preprocess_input(user_msg)
        input_vec = tfidf_vec.transform([preprocess]).toarray()  # Transform into vector form
        input_vec = input_vec.reshape((input_vec.shape[0], 1, input_vec.shape[1]))
        prediction = model.predict(input_vec,verbose=0)
#print(prediction)
#predict_class = np.argmax(prediction, axis=1)
#print(predict_class)
        if prediction[0][0]>=0.7:
            result = "This is a Ham message, so safe to use...😇💬"
    #print(result)
        else:
            result = "This is a Smish message, so avoid clicking links and be cautious!...⚠️"
        return send_file('result.html', result=result)  
    #print(result)
    #return redirect('/')
if __name__ == '__main__':
    app.run(debug=True)
