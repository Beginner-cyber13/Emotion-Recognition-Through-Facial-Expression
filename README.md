# Emotion Recognition AI

This project uses Python to detect faces and recognize emotions in real-time using your webcam.

## Prerequisites

- Python 3.11.9 (Verified)
- A webcam

## Installation

1.  The necessary libraries are listed in `requirements.txt`.
2.  Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This might have already been started by the agent)*

## Usage

Run the main script:

```bash
python main.py
```

## How it works

- **OpenCV** captures video from your webcam.
- **DeepFace** analyzes each frame to detect faces and determine the dominant emotion (happy, sad, angry, surprise, fear, disgust, neutral).
- The result is displayed on the screen with a bounding box and the emotion label.

## Note on First Run

When you run the script for the first time, DeepFace will download the pre-trained models (weights). This can take several minutes depending on your internet connection. Please be patient.
