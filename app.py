import numpy as np
from flask import Flask, request, render_template
import pickle

# Create an app object using the Flask class.
app = Flask(__name__, template_folder='template')

# Load the trained model. (Pickle file)
model1 = pickle.load(open('model.pkl', 'rb'))

# Define the route to be home.
@app.route('/')
def home():
    return render_template('index.html')

# Add Post method to the decorator to allow for form submission.
@app.route('/predict', methods=['POST'])
def predict():
    inputs = [x for x in request.form.values()]
    outputs=[]
    try:
        outputs = [float(inputs[x]) for x in range(0, 38)]
        for x in range(38, 45):
            a = inputs[x].split('/')
            a = [float(i) for i in a]
            a[0] = float(a[0]) / (8 * float(inputs[2]))
            a[1] = float(a[1]) / (8 * float(inputs[2]))
            b = abs(a[0] - a[1])
            if b >= 0.5 and b <= 1.5:
                outputs.append(0)
            else:
                outputs.append(1)
        temp = [float(inputs[x]) for x in range(45, 61)]
    except ValueError:
        return render_template('index.html',prediction_text='Please enter valid numbers.')
    
    outputs = outputs + temp
    features = [outputs]
    
    try:
        prediction = model1.predict(features)
        output = prediction[0]
    except Exception as e:
        return render_template('index.html',items=features, prediction_text=f'Error in prediction: {e}')

    return render_template('index.html', prediction_text=f'Output ==> {output}')

if __name__ == "__main__":
    app.run(debug=True)
