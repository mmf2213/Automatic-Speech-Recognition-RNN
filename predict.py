import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import extract_features

model = load_model("asr_model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))
max_len = pickle.load(open("max_len.pkl", "rb"))

def predict_audio(file):
    features = extract_features(file)

    if features is None:
        return "Invalid audio"

    features = pad_sequences(
        [features],
        maxlen=max_len,
        padding='post'
    )

    pred = model.predict(features)
    result = np.argmax(pred)

    return le.inverse_transform([result])[0]