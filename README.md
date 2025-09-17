# Bird Sound Detection using Convolutional Neural Networks

This repository contains the implementation of a deep learning system for identifying bird species from audio recordings, as detailed in the research paper "Bird Sound Detection using Convolution Neural Networks". [cite_start]The project aims to provide a scalable tool for biodiversity monitoring and ecological research[cite: 16, 20].

---

## 📋 Project Overview

[cite_start]The core of this project is a Convolutional Neural Network (CNN) that classifies bird species based on their sounds[cite: 17]. [cite_start]The system's methodology involves converting raw audio files into Mel spectrograms, which serve as visual inputs for the CNN model[cite: 17]. [cite_start]This approach allows the model to learn and distinguish the unique acoustic patterns of different bird species[cite: 18]. [cite_start]The final trained model is integrated into a web-based interface for easy user interaction[cite: 20].

### Key Features
* [cite_start]**Species Classification**: Identifies 20 distinct bird species from audio recordings[cite: 96, 186].
* [cite_start]**Deep Learning Model**: Employs a CNN to automatically extract discriminative features from sounds[cite: 17, 18].
* [cite_start]**Audio-to-Image Conversion**: Transforms audio signals into Mel spectrograms for visual pattern recognition[cite: 17].
* [cite_start]**High Accuracy**: Achieves an 84% accuracy rate on the validation dataset[cite: 19].
* [cite_start]**Web Interface**: Features a user-friendly web application built with Flask for real-time prediction[cite: 213, 217].

---

## ⚙️ Methodology

The project follows a structured workflow from data acquisition to final deployment.

1.  [cite_start]**Data Collection**: Audio recordings of 20 different bird species were gathered[cite: 96].
2.  [cite_start]**Data Preprocessing**: The audio files were converted into Mel spectrograms to serve as input for the model[cite: 17].
3.  [cite_start]**Model Selection**: A Convolutional Neural Network (CNN) was selected for its proven effectiveness in image classification tasks[cite: 17].
4.  [cite_start]**Model Training**: The CNN was trained on the dataset of spectrograms[cite: 90].
5.  [cite_start]**Model Evaluation**: The model's performance was assessed using standard metrics like accuracy and loss[cite: 91, 210].
6.  [cite_start]**Deployment**: The trained model was integrated into a web interface for practical use[cite: 93].

---

### 1. Data Collection and Preprocessing

#### Data Source
* [cite_start]The dataset was sourced from the **Xeno-Canto portal**[cite: 96].
* [cite_start]It consists of **1,872 audio recordings** in `.mp3` format, covering 20 distinct bird species[cite: 96].
* [cite_start]Each audio file includes pre-annotated segments that highlight where bird sounds are most prominent[cite: 97].

#### Mel Spectrogram Conversion
[cite_start]To prepare the data for the CNN, each audio recording was converted into a Mel spectrogram, a visual representation of sound[cite: 145]. [cite_start]This process generated a total of **20,556 Mel spectrograms**[cite: 170].

The following parameters were used for the conversion:
* [cite_start]**Sampling Rate (sr)**: 22050 Hz[cite: 160].
* [cite_start]**Window Size (n_fft)**: 2048 samples[cite: 162].
* [cite_start]**Hop Length**: 512 samples[cite: 164].
* [cite_start]**Number of Mel Bands (n_mels)**: 128[cite: 166].

---

### 2. Model Architecture

[cite_start]A Convolutional Neural Network (CNN) was designed to classify the spectrogram images[cite: 174].

* [cite_start]**Input Layer**: Accepts spectrogram images resized to `(128, 128, 1)`[cite: 176].
* [cite_start]**Convolutional Blocks**: Three sequential blocks are used for feature extraction[cite: 177].
    * [cite_start]Each block consists of a `Conv2D` layer with a `(3, 3)` filter and `ReLU` activation, followed by a `MaxPooling2D` layer with a `(2, 2)` window[cite: 177, 178, 181].
    * [cite_start]The number of filters in the convolutional layers increases from 32 to 64, and finally to 128[cite: 178, 179].
* [cite_start]**Flatten Layer**: Converts the 2D feature maps from the convolutional blocks into a 1D vector[cite: 182].
* **Dense Layers**:
    * [cite_start]A fully connected layer with 512 neurons and `ReLU` activation performs higher-level feature extraction[cite: 184].
    * [cite_start]A **Dropout Layer** with a rate of 0.5 is added immediately after to prevent overfitting[cite: 185].
* [cite_start]**Output Layer**: A final dense layer with 20 neurons (one for each species) uses a `softmax` activation function for multi-class classification[cite: 186].
* [cite_start]**Compilation**: The model is compiled using the **Adam optimizer** and the **sparse categorical cross-entropy** loss function[cite: 187].

---

### 3. Training and Evaluation

* [cite_start]**Dataset Split**: The dataset was divided into training and testing sets using an 80:20 ratio, based on the Pareto Principle[cite: 208].
* [cite_start]**Validation**: During training, the model's performance was continuously evaluated on the validation set to monitor for overfitting and ensure it generalizes well to new data[cite: 209, 211].

---

### 4. Web Interface

[cite_start]A lightweight web application was developed using the **Flask** framework to provide a user-friendly interface for the model[cite: 213].

* [cite_start]The user can upload an audio file through the web page[cite: 216].
* [cite_start]The backend server processes the audio, converts it to a Mel spectrogram, and sends it to the trained model[cite: 216, 217].
* [cite_start]The model predicts the bird's class, and the result is displayed back to the user on the website[cite: 217].

---

## 📊 Results

The model demonstrated strong and reliable performance in classifying bird species.

* [cite_start]**Training Accuracy**: The model achieved a high accuracy rate on the training dataset[cite: 19].
* [cite_start]**Validation Accuracy**: The final accuracy on the validation dataset was **84%**[cite: 19, 223].
* [cite_start]**Classification Report**: The weighted-average for precision, recall, and F1-score were all 0.84, indicating balanced performance across the different classes[cite: 224].
* [cite_start]**Confusion Matrix**: Out of 4,119 bird sounds in the testing dataset, 3,453 were predicted correctly[cite: 308].

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

## 📄 Reference

This project is an implementation of the research paper:

> D. K, C. N, M. B, J. S, and S. N. "Bird Sound Detection using Convolution Neural Networks."
