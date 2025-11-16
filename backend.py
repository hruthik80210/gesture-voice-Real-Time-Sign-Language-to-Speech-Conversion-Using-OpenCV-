from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import mediapipe as mp
from collections import deque

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load your files
model = load_model("sign_language_model.h5")
lb = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

smooth = deque(maxlen=6)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if frame is None:
        return {"prediction":"no_hand","confidence":0}

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return {"prediction":"no_hand","confidence":0}

    coords=[]
    for lm in results.multi_hand_landmarks[0].landmark:
        coords.extend([lm.x,lm.y,lm.z])

    pred = model.predict(np.array([coords]), verbose=0)
    cid = np.argmax(pred)
    cname = lb.classes_[cid]
    conf = float(np.max(pred))

    smooth.append(cname)
    final = max(set(smooth), key=smooth.count)

    return {"prediction":final,"confidence":conf}
