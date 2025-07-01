import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

#getting emotion from filename
def get_emotion(filename):

    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    emotion_code= filename.split('-')[2]
    return emotion_map.get (emotion_code,'unknown')

# Step 2: Extract MFCC features from an audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Step 3: Load all files and create dataset
def prepare_dataset(data_dir):
    features = []
    labels = []

    print("Searching in:", os.path.abspath(data_dir))

    for root, _, files in os.walk(data_dir):
        print("Checking folder:", root)
        for file in files:
            print("Found file:", file)
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print("Processing:", file_path)
                try:
                    mfcc = extract_features(file_path)
                    emotion = get_emotion(file)
                    features.append(mfcc)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    print(f"Total files processed: {len(features)}")
    return np.array(features), np.array(labels)

# Run it
X, y = prepare_dataset("data")  # <-- point to your RAVDESS dataset path
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "model/emotion_model.pkl")
print("✅ Model saved at model/emotion_model.pkl")
