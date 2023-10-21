from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classification',methods=['POST'])
def classification():
    '''
    For rendering results on HTML GUI
    '''

    # Get integer input features from a POST request and convert them to a list of integers
    int_features = [float(x) for x in request.form.values()]

    try:
        # Create a list of NumPy arrays containing the integer features
        final_features = [np.array(int_features)]
        # Use the pre-trained model to make a classification prediction on the input features
        classification = model.predict(final_features)

        # Get the predicted classification label
        output = classification[0]

        return render_template('index.html', prediction_text='NBA Players Investement should be {}'.format(output))
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter numeric values.')


if __name__ == '__main__':
    app.run(port=5000, debug=True)


