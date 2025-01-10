from flask import Flask, request, render_template
import pickle
import numpy as np


model = pickle.load(open('agri_pre.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      
        state = request.form['State_Name']
        district = request.form['District_Name']
        year = int(request.form['Crop_Year'])
        season = request.form['Season']
        crop = request.form['Crop']
        area = float(request.form['Area'])

        
        input_data = np.array([[state, district, year, season, crop, area]])
        
       
        prediction = model.predict(input_data)
        
        return f'The predicted production is: {prediction[0]} tons'

if __name__ == '__main__':
    app.run(debug=True)
