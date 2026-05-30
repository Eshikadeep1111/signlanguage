import pickle
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import os

app = Flask(__name__)
CORS(app)

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# New mediapipe API (0.10.30+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3
)

MAX_LEN = 84

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify({'prediction': None})

    data_aux = []
    for hand in result.hand_landmarks:
        x_ = [lm.x for lm in hand]
        y_ = [lm.y for lm in hand]
        for lm in hand:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

    if len(data_aux) < MAX_LEN:
        data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
    else:
        data_aux = data_aux[:MAX_LEN]

    prediction = model.predict([np.asarray(data_aux)])
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)