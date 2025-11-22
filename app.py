from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load("sentiment_model.pkl")  # Make sure filename matches exactly!

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data['Sentence']
    prediction = model.predict([sentence])[0]

    label_map = {0: ("Negative", "😞"), 1: ("Positive", "😊")}
    sentiment_text, emoji = label_map.get(int(prediction), ("Unknown", "❓"))

    return jsonify({'Sentiment': sentiment_text, 'Emoji': emoji})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
