from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'..\models\model.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging
        print("Form Data:", request.form)


        severity = int(request.form.get('severity', 0))
        age = int(request.form.get('age', 0))
        private_attorney = int(request.form.get('privateAttorney', 0))
        marital_status = request.form.get('maritalStatus', 'Unknown')
        gender = 1 if request.form.get('gender') == 'Male' else 0
        insurance = request.form.get('insurance', 'Unknown')
        specialty = request.form.get('specialty', 'Family Practice')

        # Map categorical fields to one-hot encoding
        marital_status_map = {"Married": 0, "Single": 1, "Unknown": 2, "Divorced": 3, "Widowed": 4}
        insurance_map = {
            "Medicare/Medicaid": [1, 0, 0, 0, 0],
            "No Insurance": [0, 1, 0, 0, 0],
            "Private": [0, 0, 1, 0, 0],
            "Unknown": [0, 0, 0, 1, 0],
            "Workers Compensation": [0, 0, 0, 0, 1],
        }
        specialty_map = {
            "Anesthesiology": 0, "Cardiology": 1, "Dermatology": 2,
            "Emergency Medicine": 3, "Family Practice": 4, "General Surgery": 5,
            "Internal Medicine": 6, "Neurology/Neurosurgery": 7, "OBGYN": 8,
            "Occupational Medicine": 9, "Ophthamology": 10, "Orthopedic Surgery": 11,
            "Pathology": 12, "Pediatrics": 13, "Physical Medicine": 14,
            "Plastic Surgeon": 15, "Radiology": 16, "Resident": 17,
            "Thoracic Surgery": 18, "Urological Surgery": 19,
        }

        # Encode inputs
        marital_status_encoded = marital_status_map.get(marital_status, 2)
        insurance_encoded = insurance_map.get(insurance, [0, 0, 0, 1, 0])
        specialty_encoded = [0] * len(specialty_map)
        specialty_encoded[specialty_map.get(specialty, 4)] = 1 

        # Prepare final input array
        input_features = [severity, age, private_attorney, marital_status_encoded, gender] + insurance_encoded + specialty_encoded
        final_features = np.array([input_features])

        #Debugging
        print("Final Features:", final_features)

        # Make prediction
        prediction = model.predict(final_features)[0]
 

        return jsonify({'prediction_text': f'Predicted Amount: ${prediction:,.2f}'})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
