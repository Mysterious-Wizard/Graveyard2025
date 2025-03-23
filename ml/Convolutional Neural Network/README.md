# CIFAR-10 CNN Image Classification

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is built using TensorFlow/Keras and evaluates 10 image classes including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training & Evaluation](#training--evaluation)
- [Testing the Model](#testing-the-model)
- [Project Structure](#project-structure)
- [Model Logic & Working](#model-logic--working)
- [Output](#output)
- [Future Improvements](#future-improvements)
- [Usage Instructions](#usage-instructions)
- [References & Resources](#references--resources)

---

## Overview

This project consists of two main Python scripts:
1. **train_and_save_model.py** – Loads and preprocesses the CIFAR-10 dataset, builds and trains a CNN, evaluates the model (including visualizations), and saves the trained model.
2. **test_model.py** – Loads the saved model and provides an interface to evaluate custom images by preprocessing them, predicting the class, and displaying the results.

---

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV (`opencv-python`)

Install dependencies using pip:

    ```bash
      pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python
## Setup
Clone or download this repository to your local machine.

Ensure dependencies are installed.

The project includes the following files:
    ``` bash
    
    CNN_model_train.py
    
    test_CNN.py
    
    (This README.md)
    
    a sample image for testing 

## Training & Evaluation
_The training script (train_and_save_model.py) performs the following steps:_

### Load & Preprocess Data:

- Loads CIFAR-10 from Keras.

- Normalizes pixel values to [0, 1].

- Converts labels to one-hot encoding.

- Displays a sample of images and the class distribution.

### Build CNN Architecture:

- Uses multiple convolutional layers with ReLU activation.

- Implements max pooling and dropout layers for regularization.

- Includes dense layers to output predictions for 10 classes.

### Train the Model:

- Trains the model for 15 epochs using a batch size of 64.

- Uses validation data for monitoring performance.

- Evaluate & Visualize Performance:

- Plots accuracy and loss curves for both training and validation sets.

- Computes and displays a confusion matrix.

- Outputs a classification report.

### Save the Model:

*_Saves the trained model in the saved_model directory as cifar10_cnn_model.h5._*

### Testing the Model
*The testing script (test_model.py) allows you to:*

- Load the Saved Model: Loads the model saved during training.

- Preprocess a Custom Image:

- Reads an image using OpenCV.

- Converts the image from BGR to RGB.

- Resizes it to 32x32 pixels (CIFAR-10 size).

- Normalizes the image.

**Predict the Class:**

- Predicts the class for the input image.

- Outputs the predicted class with the confidence score.

**Display the Image:**

Shows the image with the prediction overlay.

Usage from the command line:

bash
Copy
Edit
python test_model.py path/to/your/image.jpg
Project Structure
bash
Copy
Edit
├── README.md
├── train_and_save_model.py  # Training, evaluation, and model saving script
├── test_model.py            # Script to test the saved model on custom images
└── saved_model/             # Directory where the trained model is saved
Model Logic & Working
Data Loading: The model uses the CIFAR-10 dataset which consists of 60,000 images across 10 classes.

CNN Architecture:

Convolutional Layers: Extract features from the input images.

MaxPooling Layers: Reduce the spatial dimensions and help in translation invariance.

Dropout Layer: Helps to prevent overfitting by randomly disabling neurons during training.

Dense Layers: Process the flattened features to output a prediction across 10 classes.

Training Process: Uses the Adam optimizer and categorical cross-entropy loss. The training is monitored using accuracy and loss metrics on both training and validation sets.

Evaluation: Generates accuracy/loss plots, confusion matrix, and a classification report to assess model performance.

Saving & Testing: The model is saved in HDF5 format (.h5) and later loaded to evaluate custom images.

Output
Placeholder for Output:

Training Plots: Accuracy and Loss curves.

Confusion Matrix: Heatmap visualization.

Classification Report: Precision, recall, and F1-score per class.

Test Prediction: Displayed image with predicted class and confidence.

(After running the scripts, paste your outputs here for documentation.)

Future Improvements
Data Augmentation: Improve robustness by augmenting the training data.

Hyperparameter Tuning: Experiment with learning rate, dropout rate, and model depth.

Transfer Learning: Integrate pre-trained models for improved performance.

Expanding Classes: Modify the architecture/dataset for additional image classes.

Usage Instructions
Training the Model:

Run the training script:

bash
Copy
Edit
python train_and_save_model.py
This script trains the CNN, displays evaluation plots, and saves the model in the saved_model directory.

Testing the Model:

Provide an image path to the test script:

bash
Copy
Edit
python test_model.py path/to/your/image.jpg
The script will display the image with the predicted class and confidence score.

Reviewing Output:

Check the terminal and generated plots for performance metrics and evaluation results
