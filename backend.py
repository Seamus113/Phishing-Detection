from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict

# create flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict_url", methods=["POST"])
def predict_url():
    # parse request json
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "missing field 'url'"}), 400

    url = data["url"]

    try:
        # call your prediction function
        # expect: label, prob, features_dict
        label, prob, features = predict(url)
    except Exception as e:
        # basic error handling
        return jsonify({"error": str(e)}), 500

    # build response json
    return jsonify({
        "url": url,
        "label": int(label),                      # 0 = legitimate, 1 = phishing
        "probability_phishing": float(prob),      # model confidence
        "features": features                      # optional, for debugging / UI
    })


if __name__ == "__main__":
    # dev mode server
    app.run(host="0.0.0.0", port=5000, debug=True)
