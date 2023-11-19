from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('arima_alcohol_insgesamt.pkl', 'rb') as file:
    arima_alcohol_insgesamt = pickle.load(file)

with open('arima_alcohol_killed.pkl', 'rb') as file:
    arima_alcohol_killed = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.Series(data['input_data'])
        forecast_values_insgesamt = arima_alcohol_insgesamt.predict(start=len(arima_alcohol_insgesamt.fittedvalues), end=len(arima_alcohol_insgesamt.fittedvalues) + len(input_data) - 1, dynamic=False)
        forecast_values_killed = arima_alcohol_killed.predict(start=len(arima_alcohol_killed.fittedvalues), end=len(arima_alcohol_killed.fittedvalues) + len(input_data) - 1, dynamic=False)

        forecast_values_insgesamt = forecast_values_insgesamt.tolist()
        forecast_values_killed = forecast_values_killed.tolist()

        return jsonify({'forecast_values_insgesamt': forecast_values_insgesamt,
                        'forecast_values_killed': forecast_values_killed})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
