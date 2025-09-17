# Bird Sound Detection using Convolutional Neural Networks 🐦🔊

This repository contains the implementation of a deep learning system for identifying bird species from audio recordings, as detailed in the research paper "Bird Sound Detection using Convolution Neural Networks". The project aims to provide a scalable tool for biodiversity monitoring and ecological research.

---

## 📋 Project Overview

The core of this project is a Convolutional Neural Network (CNN) that classifies bird species based on their sounds. The system's methodology involves converting raw audio files into Mel spectrograms, which serve as visual inputs for the CNN model. This approach allows the model to learn and distinguish the unique acoustic patterns of different bird species. The final trained model is integrated into a web-based interface for easy user interaction.

### Key Features
* **Species Classification**: Identifies 20 distinct bird species from audio recordings.
* **Deep Learning Model**: Employs a CNN to automatically extract discriminative features from sounds.
* **Audio-to-Image Conversion**: Transforms audio signals into Mel spectrograms for visual pattern recognition.
* **High Accuracy**: Achieves an 84% accuracy rate on the validation dataset.
* **Web Interface**: Features a user-friendly web application built with Flask for real-time prediction.

---

## ⚙️ Methodology

The project follows a structured workflow from data acquisition to final deployment.

1.  **Data Collection**: Audio recordings of 20 different bird species were gathered.
2.  **Data Preprocessing**: The audio files were converted into Mel spectrograms to serve as input for the model.
3.  **Model Selection**: A Convolutional Neural Network (CNN) was selected for its proven effectiveness in image classification tasks.
4.  **Model Training**: The CNN was trained on the dataset of spectrograms.
5.  **Model Evaluation**: The model's performance was assessed using standard metrics like accuracy and loss.
6.  **Deployment**: The trained model was integrated into a web interface for practical use.

---

### 1. Data Collection and Preprocessing

#### Data Source
* The dataset was sourced from the **Xeno-Canto portal**.
* It consists of **1,872 audio recordings** in `.mp3` format, covering 20 distinct bird species.
* Each audio file includes pre-annotated segments that highlight where bird sounds are most prominent.

#### Mel Spectrogram Conversion
To prepare the data for the CNN, each audio recording was converted into a Mel spectrogram, a visual representation of sound. This process generated a total of **20,556 Mel spectrograms**.

The following parameters were used for the conversion:
* **Sampling Rate (sr)**: 22050 Hz.
* **Window Size (n_fft)**: 2048 samples.
* **Hop Length**: 512 samples.
* **Number of Mel Bands (n_mels)**: 128.

---

### 2. Model Architecture

A Convolutional Neural Network (CNN) was designed to classify the spectrogram images.

* **Input Layer**: Accepts spectrogram images resized to `(128, 128, 1)`.
* **Convolutional Blocks**: Three sequential blocks are used for feature extraction.
    * Each block consists of a `Conv2D` layer with a `(3, 3)` filter and `ReLU` activation, followed by a `MaxPooling2D` layer with a `(2, 2)` window.
    * The number of filters in the convolutional layers increases from 32 to 64, and finally to 128.
* **Flatten Layer**: Converts the 2D feature maps from the convolutional blocks into a 1D vector.
* **Dense Layers**:
    * A fully connected layer with 512 neurons and `ReLU` activation performs higher-level feature extraction.
    * A **Dropout Layer** with a rate of 0.5 is added immediately after to prevent overfitting.
* **Output Layer**: A final dense layer with 20 neurons (one for each species) uses a `softmax` activation function for multi-class classification.
* **Compilation**: The model is compiled using the **Adam optimizer** and the **sparse categorical cross-entropy** loss function.

---

### 3. Training and Evaluation

* **Dataset Split**: The dataset was divided into training and testing sets using an 80:20 ratio, based on the Pareto Principle.
* **Validation**: During training, the model's performance was continuously evaluated on the validation set to monitor for overfitting and ensure it generalizes well to new data.

---

### 4. Web Interface

A lightweight web application was developed using the **Flask** framework to provide a user-friendly interface for the model.

* The user can upload an audio file through the web page.
* The backend server processes the audio, converts it to a Mel spectrogram, and sends it to the trained model.
* The model predicts the bird's class, and the result is displayed back to the user on the website.

---

## 📊 Results

The model demonstrated strong and reliable performance in classifying bird species.

* **Training Accuracy**: The model achieved an accuracy of 90.17% on the training set.
* **Validation Accuracy**: The final accuracy on the validation dataset was 84%.
* **Classification Report**: The weighted-average for precision, recall, and F1-score were all 0.84, indicating balanced performance across the different classes.
* **Confusion Matrix**: Out of 4,119 bird sounds in the testing dataset, 3,453 were predicted correctly.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Pip

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/bird-sound-detection.git](https://github.com/your-username/bird-sound-detection.git)
    cd bird-sound-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```

5.  Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

---
