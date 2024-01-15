from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

# Load the feature label encoders
label_encoders = {
    'arrival_date_month': {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                           'September': 9, 'October': 10, 'November': 11, 'December': 12},
    'meal': {'BB': 1, 'FB': 2, 'HB': 3, 'SC': 4},
    'reserved_room_type': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'L': 9, 'P': 10},
    'customer_type': {'Transient': 1, 'Contract': 2, 'Transient-Party': 3, 'Group': 4},
    'reservation_status': {'Check-Out': 1, 'Canceled': 2, 'No-Show': 3}
}

# Load the HTML form page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []

        for feature_name in label_encoders.keys():
            selected_option = request.form[feature_name]
            features.append(label_encoders[feature_name][selected_option])

        for feature_name in model.feature_names:
            if feature_name not in label_encoders:
                features.append(float(request.form[feature_name]))

        # Make the prediction
        prediction = model.predict([features])[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error='Error in prediction: {}'.format(e))

if __name__ == '__main__':
    app.run(debug=True)
