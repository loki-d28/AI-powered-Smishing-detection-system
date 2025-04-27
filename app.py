from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Load model and vectorizer
try:
    model = load_model('smish_model.h5')
    with open('tfidf_vec.pkl', 'rb') as f:
        tfidf_vec = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

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
        try:
            user_msg = request.form.get('message')
            
            # Process the input
            preprocess = preprocess_input(user_msg)
            input_vec = tfidf_vec.transform([preprocess]).toarray()
            input_vec = input_vec.reshape((input_vec.shape[0], 1, input_vec.shape[1]))
            prediction = model.predict(input_vec, verbose=0)
            
            # Determine result
            if prediction[0][0] >= 0.7:
                result = "This is a Ham message, so safe to use...😇💬"
                result_color = "green"
            else:
                result = "This is a Smish message, so avoid clicking links and be cautious!...⚠️"
                result_color = "red"
            
            # Render template with result and color
            return render_template('result.html', result=result, result_color=result_color)
        
        except Exception as e:
            return f"Error processing your request: {e}"
    
    # If not POST request, redirect to home
    return redirect('/')

# Configure for production
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
