import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# MediaPipe (static mode since we get individual frames)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

MAX_LEN = 84

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image from browser
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({'prediction': None})

    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

    if len(data_aux) < MAX_LEN:
        data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
    else:
        data_aux = data_aux[:MAX_LEN]

    prediction = model.predict([np.asarray(data_aux)])
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    # 0.0.0.0 makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=False)