import os
import numpy as np
import pickle

from utils import extract_features

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

X = []
y = []

data_path = "dataset"
max_samples = 10000

for label in os.listdir(data_path):
    label_path = os.path.join(data_path, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):

        if len(X) >= max_samples:
            break

        file_path = os.path.join(label_path, file)

        try:
            features = extract_features(file_path)

            if features is None:
                continue

            X.append(features)
            y.append(label)

        except Exception:
            continue

    if len(X) >= max_samples:
        break

print("Total samples loaded:", len(X))

X = pad_sequences(X, padding='post', dtype='float32')

# 🔥 SAVE MAX LENGTH (IMPORTANT)
max_len = X.shape[1]
pickle.dump(max_len, open("max_len.pkl", "wb"))

le = LabelEncoder()
y = le.fit_transform(y)

pickle.dump(le, open("label_encoder.pkl", "wb"))

input_layer = Input(shape=(X.shape[1], X.shape[2]))

x = LSTM(64)(input_layer)

output = Dense(len(set(y)), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X,
    y,
    epochs=10,
    batch_size=16,
    validation_split=0.2
)

model.save("asr_model.h5")

print("Training completed successfully!")