# Real-Time Gender Detection via Computer Vision

A real-time gender classification system built with a custom-trained Convolutional Neural Network (CNN) using Tensorflow Keras and OpenCV. The system detects faces via webcam and classifies gender with **~95% validation accuracy**.

---

## Demo

> Live webcam feed with bounding boxes and gender predictions rendered in real time.

<p text-align: center>
  <img width="450" height="356" alt="image" src="https://github.com/user-attachments/assets/36bcf7a9-76cc-4b85-8791-7517aabdb8de" />
</p>



## Features

- **~95% validation accuracy** on held-out dataset
- **Real-time inference** via webcam using OpenCV
- **Custom CNN** trained from scratch — no pretrained backbone
- **Haar Cascade face detection** to isolate and crop faces before classification
- **Confidence scores** displayed alongside predictions
- **Data augmentation** pipeline to improve generalization

---
## How to use

1)
```bash
pip install tensorflow opencv-python numpy matplotlib
```
2) download gender_detection_CV.py
3) download gender_detection_model.keras
4) download haarcascade_frontalface_alt.xml
5) change file path on line 12 to saved location of gender_detection_model.keras
6) change file path on line 15 to saved location of haarcascade_frontalface_alt.xml
7) run gender_detection_CV

note: if you are having trouble use an environment and install dependencies from step 1 in the environment, i recommend miniconda if you don't already have an environment

---

## Model Architecture

The CNN was designed and trained from scratch with a deliberate architecture to balance performance and overfitting prevention:

| Layer | Details |
|---|---|
| Input | 128 × 128 × 3 (RGB) |
| Data Augmentation | Random flip, rotation, zoom, brightness, contrast |
| Rescaling | Normalize pixel values to [0, 1] |
| Conv2D + BN + MaxPool | 16 filters, 3×3 kernel |
| Conv2D + BN + MaxPool | 32 filters, 3×3 kernel |
| Conv2D + BN + MaxPool | 64 filters, 3×3 kernel |
| Dense + Dropout (0.5) | 128 units, ReLU |
| Dense + Dropout (0.3) | 64 units, ReLU |
| Output | 2 units, Softmax |

**Optimizer:** Adam  
**Loss:** Sparse Categorical Cross-Entropy  
**Regularization:** Batch Normalization + Dropout  
**Early Stopping:** Monitors `val_accuracy`, patience of 3 epochs

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Model training and inference |
| OpenCV | Webcam capture and face detection |
| NumPy | Array manipulation |

---

## Project Structure

```
gender-detection/
├── gender_detection_cnn.py       # Model definition, training, and saving
├── gender_detection_CV.py        # Real-time webcam inference pipeline
├── gender_detection_model.keras  # Saved trained model
├── haarcascade_frontalface_alt.xml  # OpenCV pre-trained face detector
└── README.md
```

### Prerequisites

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### Training the Model

> Requires a dataset organized into `Train/` and `Validation/` subdirectories with class-labeled folders (e.g., `Male/`, `Female/`).

```bash
python gender_detection_cnn.py
```

The trained model will be saved as `gender_detection_model.keras`.

### Running Real-Time Detection

```bash
python gender_detection_CV.py
```

- Ensure your webcam is connected
- Press **`Q`** to quit the live feed

---

## How It Works

1. **Frame Capture** — OpenCV reads frames from the webcam in real time
2. **Face Detection** — Haar Cascade classifier locates faces in each frame
3. **Preprocessing** — Detected face regions are cropped, converted to RGB, and resized to 128×128
4. **Inference** — The CNN outputs a softmax probability vector over two classes
5. **Display** — A bounding box and labeled confidence score are rendered onto the frame

---

## Training Details

- **Data Augmentation** was applied to improve robustness to lighting, orientation, and scale variations
- **Batch Normalization** was used after each convolutional block to stabilize training and reduce internal covariate shift
- **Dropout** layers (0.5 and 0.3) were placed after dense layers to prevent over-reliance on specific neurons
- **Early Stopping** halts training when validation accuracy plateaus, restoring the best-performing weights

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~95% |
| Input Resolution | 128 × 128 px |
| Inference | Real-time (webcam) |

---

## Future Improvements

- [ ] Migrate to a lightweight pretrained backbone (e.g., MobileNetV2) for higher accuracy
- [ ] Add multi-face tracking across frames
- [ ] Export to ONNX or TensorFlow Lite for edge deployment
- [ ] Build a simple GUI or web interface

---

## Author

- William Roche
- Electrical Engineering Student
- Florida Polytechnic University
- LinkedIn link in bio

Built as a personal computer vision project to explore CNN design, real-time inference pipelines, and OpenCV integration, and to improve robotics projects
