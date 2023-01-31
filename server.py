import pickle
from flask import Flask, request, jsonify

model = pickle.load(open("rating_model.pkl", "rb"))
blowback_encoder = pickle.load(open("blowback_encoder.pkl", "rb"))
fuel_encoder = pickle.load(open("fuel_encoder.pkl", "rb"))

app = Flask(__name__)


@app.route("/", methods=['POST'])
def predict():
    data = request.get_json()
    year = data['y']
    month = data['m']
    battery_value = data['bv']
    dipstick_value = data['dv']
    engine_oil = data['oil']
    engine_value = data['ev']
    coolant_value = data['coolant']
    engine_mounting_value = data['mv']
    sound_value = data['sound']
    smoke_value = data['smoke']
    compression_value = data['compression']
    blowback = blowback_encoder.transform([data['blowback']])[0]
    clutch_value = data['clutch']
    gear_shift_value = data['gear']
    fuel = fuel_encoder.transform([data['fuel']])[0]
    odometer = data['odo']
    return jsonify(int(model.predict(
        [[year, month, battery_value, dipstick_value, engine_oil, engine_value, coolant_value, engine_mounting_value,
          sound_value, smoke_value, compression_value, blowback, clutch_value, gear_shift_value, fuel, odometer]])[0]))


if __name__ == "__main__":
    app.run(debug=True)
