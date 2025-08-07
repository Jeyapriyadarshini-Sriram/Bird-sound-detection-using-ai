from flask import Flask, request, render_template
import os
import librosa
import numpy as np
from tensorflow import keras
from keras.models import load_model
import cv2
from collections import Counter

# Initialize Flask app
app = Flask(__name__)



model_path = 'models/trained_model.h5'
model = load_model(model_path)

class_labels = [
    'Acrocephalus melanopogon', 'Acrocephalus melanopogon', 'Acrocephalus scirpaceus', 'Alcedo atthis',
    'Anas platyrhynchos', 'Anas strepera', 'Ardea purpurea', 'Botaurus stellaris', 'Charadrius alexandrinus',
    'Ciconia ciconia', 'Circus aeruginosus', 'Coracias garrulus', 'Dendrocopos minor', 'Fulica atra',
    'Gallinula chloropus', 'Himantopus himantopus', 'Ixobrychus minutus', 'Motacilla flava', 'Porphyrio porphyrio',
    'Tachybaptus ruficollis'
]

def load_mel_spectrogram(file_path):
    return np.load(file_path)

# Function to preprocess data for prediction
def preprocess_data(spectrogram, target_size=(128, 128)):
    # Resize spectrogram
    resized_spectrogram = cv2.resize(spectrogram, target_size)
    # Add extra dimensions to match model input shape
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=0)
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
    return resized_spectrogram

# Function to make predictions
def predict(model, spectrogram):
    # Preprocess the spectrogram
    preprocessed_spectrogram = preprocess_data(spectrogram)
    # Make prediction
    prediction = model.predict(preprocessed_spectrogram)
    return prediction

def generate_mel_spectrogram(audio_path, output_dir, model, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Calculate total number of seconds
    total_seconds = len(y) // sr

    # Process each second of the audio
    for sec in range(total_seconds):
        start_frame = int(sec * sr)
        end_frame = int((sec + 1) * sr)

        # Check if segment length is long enough for n_fft
        if end_frame - start_frame >= n_fft:
            # Extract segment
            segment = y[start_frame:end_frame]

            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)

            # Save spectrogram as .npy
            output_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{sec}.npy"
            np.save(os.path.join(output_dir, output_filename), S_db)

            # Load the saved spectrogram
            mel_spectrogram = load_mel_spectrogram(os.path.join(output_dir, output_filename))

            # Make predictions
            prediction = predict(model, mel_spectrogram)
            predicted_label = np.argmax(prediction)
            predicted_class = class_labels[predicted_label]



def process_single_audio(audio_path, output_dir, model):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionary to store predictions for each second
    predictions = {}

    # Generate mel spectrograms for the single audio file
    generate_mel_spectrogram(audio_path, output_dir, model)

    # Iterate through generated spectrograms and aggregate predictions
    for file in os.listdir(output_dir):
        if file.endswith('.npy'):
            mel_spectrogram = load_mel_spectrogram(os.path.join(output_dir, file))
            prediction = predict(model, mel_spectrogram)
            predicted_label_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_label_index]

            # Update predictions dictionary
            predictions[file] = predicted_label

    # Count occurrences of each predicted label
    label_counts = Counter(predictions.values())

    # Determine the most predicted label
    final_prediction = max(label_counts, key=label_counts.get)

    return final_prediction


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        audio_file = request.files['file']
        audio_path = 'F:/audiopredicttemp/'
        audio_file.save(audio_path)

        output_dir = 'F:/spectrogrampredict/'
        pred = process_single_audio(audio_path, output_dir, model)

        # Get final prediction
        final_prediction = pred

        for dirpath, dirnames, filenames in os.walk(audio_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)

        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)

        return render_template('index.html', prediction=final_prediction)
        

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)