# Yoga Pose Classification

A deep learning project for classifying yoga poses from images. This repository contains two complementary approaches to solve the same multi-class image classification problem: a **PyTorch transfer learning model** using GoogLeNet and a **TensorFlow/Keras custom CNN** with hyperparameter tuning and real-time webcam prediction capabilities.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebook 1: PyTorch GoogLeNet Transfer Learning](#notebook-1-pytorch-googlenet-transfer-learning)
- [Notebook 2: TensorFlow/Keras Custom CNN](#notebook-2-tensorflowkeras-custom-cnn)
- [Results Comparison](#results-comparison)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

---

## Overview

The business objective of this project is to provide an accurate and reliable tool for yoga practitioners to track their progress and ensure that they are performing poses correctly. This can help reduce the risk of injury and improve the effectiveness of yoga practice.

The target audience includes individuals who practice yoga regularly, yoga teachers, fitness professionals, and anyone who wants to improve their yoga skills.

**Supported Yoga Poses (5 Classes):**
| Class | Description |
|-------|-------------|
| `downdog` | Downward-Facing Dog pose |
| `goddess` | Goddess pose |
| `plank` | Plank pose |
| `tree` | Tree pose |
| `warrior2` | Warrior II pose |

---

## Dataset

The dataset is organized into training and test sets with the following distribution:

| Split | Images | downdog | goddess | plank | tree | warrior2 |
|-------|--------|---------|---------|-------|------|----------|
| **Train** | 989 | 223 | 178 | 264 | 159 | 250 |
| **Test** | 420 | 97 | 77 | 114 | 69 | 109 |

- **Total images:** 1,409
- **Number of classes:** 5
- **Image format:** JPG
- **Data source:** Images are loaded from a cleaned dataset directory structure with class-named folders

### Data Preprocessing

Both notebooks apply standard image preprocessing:
- Image resizing and normalization
- Pixel value scaling (ImageNet statistics for PyTorch, [0,1] range for TensorFlow)
- Label encoding for multi-class classification

---

## Project Structure

```
yoga-pose-classification/
├── README.md                            # This documentation file
├── notebook_pytorch_googlenet.ipynb     # PyTorch transfer learning approach
├── notebook_tensorflow_cnn.ipynb        # TensorFlow custom CNN approach
└── dataset/
    ├── TRAIN/                           # Training images (989 images)
    │   ├── downdog/
    │   ├── goddess/
    │   ├── plank/
    │   ├── tree/
    │   └── warrior2/
    └── TEST/                            # Test images (420 images)
        ├── downdog/
        ├── goddess/
        ├── plank/
        ├── tree/
        └── warrior2/
```

---

## Notebook 1: PyTorch GoogLeNet Transfer Learning

This notebook implements a yoga pose classifier using **transfer learning with GoogLeNet (Inception v1)**, a powerful CNN architecture pre-trained on ImageNet.

### Key Features

- **Framework:** PyTorch
- **Architecture:** GoogLeNet (pretrained on ImageNet)
- **Transfer Learning Strategy:** Fine-tuning all layers with a replaced final fully-connected layer
- **Input Size:** 224 x 224 pixels

### Data Augmentation (Training Set)

```python
transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Pretrained Model | GoogLeNet |
| Input Features (FC) | 1,024 |
| Output Classes | 5 |
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD (lr=0.001, momentum=0.9) |
| Training Epochs | 25 |
| Batch Size (Train) | 64 |
| Batch Size (Validation) | 32 |
| Batch Size (Test) | 32 |
| Train/Validation Split | 80% / 20% |
| GPU Support | Yes (CUDA) |

### Training Pipeline

1. **Data Loading:** Images loaded using `torchvision.datasets.ImageFolder`
2. **Data Splitting:** Training set split into 80% train / 20% validation using `SubsetRandomSampler`
3. **Model Setup:** Pretrained GoogLeNet loaded, final FC layer replaced for 5-class output
4. **Training Loop:** Custom training function with validation loss monitoring
5. **Model Checkpointing:** Best model saved based on minimum validation loss
6. **Evaluation:** Test accuracy computed on held-out test set
7. **Visualization:** Sample predictions displayed with color-coded labels (green=correct, red=incorrect)

### Test Results

| Metric | Value |
|--------|-------|
| **Test Loss** | 0.124343 |
| **Test Accuracy** | **97% (453/466 correct)** |

---

## Notebook 2: TensorFlow/Keras Custom CNN

This notebook builds a **custom Convolutional Neural Network from scratch** using TensorFlow/Keras, with extensive hyperparameter tuning and real-time webcam-based prediction capabilities.

### Key Features

- **Framework:** TensorFlow/Keras
- **Architecture:** Custom CNN with 5 convolutional blocks
- **Hyperparameter Tuning:** KerasTuner (RandomSearch & Hyperband)
- **Input Size:** 150 x 150 pixels
- **Real-time Prediction:** OpenCV webcam integration for live pose classification

### Model Architecture

```
Input (150, 150, 3)
  |
  Conv2D(32) -> AveragePooling2D -> BatchNormalization
  Conv2D(32) -> AveragePooling2D
  Conv2D(96) -> AveragePooling2D
  Conv2D(128) -> MaxPooling2D
  Conv2D(192) -> MaxPooling2D
  |
  Flatten
  |
  Dense(512) -> Dropout(0.5)
  |
  Dense(5, activation='softmax')
```

**Total Parameters:** 1,946,053 (1,945,989 trainable, 64 non-trainable)

### Hyperparameter Tuning

The notebook explores optimal configurations using KerasTuner:

| Tuned Parameter | Search Space |
|-----------------|--------------|
| Conv1 Filters | 32 - 64 (step 16) |
| Conv2 Filters | 32 - 64 (step 16) |
| Conv3 Filters | 64 - 128 (step 32) |
| Conv4 Filters | 64 - 128 (step 32) |
| Conv5 Filters | 128 - 256 (step 64) |
| Dense Units | 256 - 1024 (step 256) |
| Dropout Rate | 0.1 - 0.5 (step 0.1) |
| Number of Layers | 1, 2, 3, 4, 5 |

**Tuning Methods:**
- `RandomSearch` for parameter exploration
- `Hyperband` for efficient resource allocation during search

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam (learning_rate=0.0001) |
| Training Epochs | 15 (with EarlyStopping) |
| Batch Size | 128 |
| Early Stopping Patience | 4 epochs |
| Callbacks | EarlyStopping, ModelCheckpoint |

### Training Pipeline

1. **Data Loading:** Custom `load_data()` function using OpenCV, images resized to (150, 150)
2. **Preprocessing:** Pixel normalization (divide by 255), labels converted to categorical
3. **EDA:** Class distribution visualization (bar charts and pie charts)
4. **Hyperparameter Search:** KerasTuner to find optimal architecture
5. **Model Training:** Final model trained with callbacks for early stopping and checkpointing
6. **Evaluation:** Test accuracy, prediction visualization, true/false prediction counts
7. **Deployment:** Webcam-based real-time pose prediction

### Test Results

| Metric | Value |
|--------|-------|
| **Test Loss** | 0.5334 |
| **Test Accuracy** | **~85.4% (398/466 correct, 68 incorrect)** |

### Real-Time Webcam Prediction

The notebook includes a complete webcam integration pipeline:

1. **Capture:** Opens default camera with a 10-second countdown timer
2. **Screenshot:** Automatically saves a captured frame as `screenshot.png`
3. **Preprocessing:** Resizes image to (150, 150), normalizes pixel values
4. **Inference:** Loads the trained model and predicts the pose class
5. **Output:** Displays the predicted yoga pose name

```python
# Load and predict on a captured image
image = cv2.imread('screenshot.png')
image = cv2.resize(image, (150, 150))
image = np.expand_dims(image, axis=0)
image = image / 255.0

prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print(class_names[predicted_class])  # e.g., "plank"
```

---

## Results Comparison

| Aspect | PyTorch GoogLeNet | TensorFlow Custom CNN |
|--------|-------------------|----------------------|
| **Framework** | PyTorch | TensorFlow/Keras |
| **Approach** | Transfer Learning | Custom CNN from scratch |
| **Architecture** | GoogLeNet (pretrained) | 5-block custom CNN |
| **Input Size** | 224 x 224 | 150 x 150 |
| **Test Accuracy** | **97%** | **~85.4%** |
| **Parameters** | ~6.8M (pretrained) | ~1.95M |
| **Training Time** | 25 epochs | 15 epochs + tuning |
| **Augmentation** | Rotation, Crop, Flip | None (tuning-focused) |
| **Webcam Support** | No | Yes |
| **Hyperparameter Tuning** | Manual | KerasTuner (automated) |

### Key Takeaways

- **GoogLeNet Transfer Learning** achieves significantly higher accuracy (97%) by leveraging pretrained ImageNet features, making it ideal for production use where accuracy is paramount.
- **Custom CNN** offers more flexibility and includes advanced features like hyperparameter tuning and real-time webcam prediction, making it better suited for experimentation and interactive applications.

---

## Requirements

### Notebook 1 (PyTorch)

```
torch
torchvision
numpy
pandas
matplotlib
```

### Notebook 2 (TensorFlow)

```
tensorflow
keras
tensorflow.keras
keras-tuner
numpy
pandas
matplotlib
opencv-python
Pillow
tqdm
```

### Full Environment

```bash
pip install torch torchvision tensorflow keras-tuner numpy pandas matplotlib opencv-python Pillow tqdm
```

---

## Usage

### Running the PyTorch Notebook

```bash
jupyter notebook notebook_pytorch_googlenet.ipynb
```

1. Update the dataset paths to point to your local `TRAIN` and `TEST` directories
2. Run all cells sequentially
3. The best model will be saved automatically based on validation loss
4. Final test accuracy and prediction visualizations will be displayed

### Running the TensorFlow Notebook

```bash
jupyter notebook notebook_tensorflow_cnn.ipynb
```

1. Update the dataset paths in the `load_data()` function
2. Run all cells sequentially
3. The hyperparameter tuning section may take significant time (~50 minutes)
4. The best model will be saved to the `models/` directory
5. For webcam prediction, ensure you have a working camera connected

### Dataset Path Configuration

**PyTorch notebook:**
```python
path_t = 'path/to/DATASET/TEST/'
path = 'path/to/DATASET/TRAIN/'
```

**TensorFlow notebook:**
```python
datasets = [
    'path/to/DATASET/TRAIN',
    'path/to/DATASET/TEST'
]
```

---

## License

This project is intended for educational and research purposes. Please ensure you have the appropriate rights to use the dataset and respect the licenses of the underlying frameworks (PyTorch, TensorFlow) and pretrained models.

---

## Acknowledgments

- **PyTorch Team** for the torchvision pretrained models
- **TensorFlow Team** for Keras and KerasTuner
- The GoogLeNet (Inception) architecture: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), Szegedy et al.
