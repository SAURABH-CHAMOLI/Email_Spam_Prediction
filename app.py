from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('lr.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Label Encoder & Feature Extraction
le = pickle.load(open('Label_Encoder.pkl', 'rb'))
feature_extraction=pickle.load(open("Feature_Extraction.pkl",'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data=request.form.get('Email_Message')

    print(data)
    

    # Feature Extraction
    input_mail_feature=feature_extraction.transform([data])

    # Make a prediction
    prediction = model.predict(input_mail_feature)

    # Return the result
    if(prediction[0]==0):
        return render_template('index.html', prediction_text="Ham Mail")
    else :
        return render_template('index.html', prediction_text="Spam Mail")
    

if __name__ == "__main__":
    app.run(debug=True)