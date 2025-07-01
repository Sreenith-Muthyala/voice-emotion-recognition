# voice-emotion-recognition
a real-time application that can predict the emotion from a short audio recording using machine learning and audio signal processing.

What the app does:
✅ Takes a .wav file as input
✅ Extracts MFCC audio features with librosa
✅ Uses a Random Forest classifier trained on the RAVDESS dataset
✅ Classifies emotions like happy, sad, angry, fearful, and more
✅ Displays predictions instantly through an interactive Streamlit web app

⚙️ Tech Stack:

Python (NumPy, scikit-learn, librosa)

Streamlit for the frontend

Joblib for model serialization

RAVDESS dataset for training

💡 Key Learnings:

Feature engineering with MFCCs

Handling audio data pipelines

Building end-to-end ML apps with deployment-ready workflows

Interpreting model performance and limitations with real-world audio inputs
