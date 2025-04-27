from flask import Flask, request, render_template, send_file, redirect, url_for, make_response
import re
import pickle
import nltk
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources only once at startup
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Global variables for model and vectorizer
model = None
tfidf_vec = None

# Load model and vectorizer
def load_ml_components():
    global model, tfidf_vec
    try:
        model = load_model('smish_model.h5')
        with open('tfidf_vec.pkl', 'rb') as f:
            tfidf_vec = pickle.load(f)
        print("Model and vectorizer loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Preprocessing input text function
def preprocess_input(text):
    result = re.sub('[^a-zA-Z]', ' ', text)
    result = result.lower()
    result = result.split()
    stop_words = set(stopwords.words('english'))
    result = ' '.join([word for word in result if word not in stop_words])
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
            # Make sure model is loaded
            global model, tfidf_vec
            if model is None or tfidf_vec is None:
                success = load_ml_components()
                if not success:
                    return "Error loading the machine learning components. Please try again later."
            
            user_msg = request.form.get('message')
            if not user_msg:
                return redirect('/')
            
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
            
            # Use direct HTML response instead of template rendering
            html_content = f'''
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <title>Smishing Detector - Result</title>
              <style>
                body {{
                  background: #f0f2f5;
                  font-family: 'Segoe UI', sans-serif;
                  display: flex;
                  justify-content: center;
                  align-items: center;
                  height: 100vh;
                  margin: 0;
                }}
                .container {{
                  background: white;
                  padding: 40px;
                  border-radius: 12px;
                  box-shadow: 0 0 20px rgba(0,0,0,0.1);
                  width: 100%;
                  max-width: 500px;
                  text-align: center;
                }}
                h1 {{
                  margin-bottom: 10px;
                  color: #2c3e50;
                }}
                p {{
                  color: #555;
                  margin-bottom: 20px;
                }}
                .result {{
                  font-size: 20px;
                  font-weight: bold;
                  color: {result_color};
                  margin-top: 30px;
                }}
                a {{
                  display: inline-block;
                  margin-top: 20px;
                  text-decoration: none;
                  color: white;
                  background-color: #3498db;
                  padding: 10px 20px;
                  border-radius: 8px;
                  transition: background 0.3s ease;
                }}
                a:hover {{
                  background-color: #2980b9;
                }}
              </style>
            </head>
            <body>
              <div class="container">
                <h1>Smishing Detector</h1>
                <p>Your message has been analyzed. Here's the result:</p>
                <div class="result">{result}</div>
                <a href="/">🔁 Check Another Message</a>
              </div>
            </body>
            </html>
            '''
            
            response = make_response(html_content)
            response.headers['Content-Type'] = 'text/html'
            return response
        
        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            print(error_message)
            return error_message
    
    # If not POST request, redirect to home
    return redirect('/')

# Ensure model is loaded when app starts
@app.before_first_request
def before_first_request():
    load_ml_components()

# Configure for production
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
