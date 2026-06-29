# Brain Tumor Classification using EfficientNetB0 and Transfer Learning

## Project Description
Brain tumors are abnormal growths of cells inside the brain that can be life-threatening if not detected at an early stage. MRI (Magnetic Resonance Imaging) scans are widely used by medical professionals to identify different types of brain tumors. Manual diagnosis of MRI images can be time-consuming and requires expert knowledge.

This project uses **Deep Learning** and **Transfer Learning** with the **EfficientNetB0** model to automatically classify brain MRI images into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

The model is trained using a publicly available MRI dataset and evaluated using various performance metrics such as accuracy, confusion matrix, and classification report.

---

## Objectives

- Load and preprocess brain MRI images.
- Perform data augmentation to improve model generalization.
- Build a brain tumor classification model using EfficientNetB0.
- Train and fine-tune the model using transfer learning.
- Evaluate the model using test accuracy, confusion matrix, and classification report.
- Save the trained model for future predictions.
- Predict the class of a new MRI image.

---

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## Dataset

Brain Tumor MRI Dataset (Kaggle)

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

Source code:
# ==========================================================
# SECTION 2: IMPORT REQUIRED LIBRARIES
# ==========================================================
# This section imports all the libraries required for
# loading the dataset, preprocessing images, building
# the deep learning model, training, evaluation,
# visualization, and prediction.
# ==========================================================

# TensorFlow library for deep learning
import tensorflow as tf

# NumPy for numerical computations
import numpy as np

# Matplotlib for plotting graphs and displaying images
import matplotlib.pyplot as plt

# OS library for file and directory operations
import os

# Load images directly from folders
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Build neural network models
from tensorflow.keras import layers, models

# Import EfficientNetB0 for Transfer Learning
from tensorflow.keras.applications import EfficientNetB0

# Import callbacks used during training
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# Import evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report

# Import Seaborn for Confusion Matrix visualization
import seaborn as sns

# Import image utilities for prediction
from tensorflow.keras.preprocessing import image

# Import model loading function
from tensorflow.keras.models import load_modelnext
# ==========================================================
# SECTION 3: MOUNT GOOGLE DRIVE
# ==========================================================
# Mount Google Drive to access the dataset stored in your
# personal Google Drive. After mounting, all files inside
# the drive can be accessed using their file paths.
# ==========================================================

from google.colab import drive

drive.mount('/content/drive')
# ==========================================================
# SECTION 4: DEFINE DATASET PATHS
# ==========================================================
# Specify the paths to the training and testing dataset
# folders stored in Google Drive.
#
# Dataset Structure:
#
# archive (2)
# ├── Training
# │   ├── glioma
# │   ├── meningioma
# │   ├── notumor
# │   └── pituitary
# │
# └── Testing
#     ├── glioma
#     ├── meningioma
#     ├── notumor
#     └── pituitary
# ==========================================================

train_path = "/content/drive/MyDrive/archive (2)/Training"
test_path = "/content/drive/MyDrive/archive (2)/Testing"
# ==========================================================
# SECTION 5: DEFINE IMAGE PARAMETERS
# ==========================================================
# IMG_SIZE:
# Images are resized to 224 × 224 pixels because
# EfficientNetB0 expects this input size.
#
# BATCH_SIZE:
# Number of images processed together during one
# forward and backward pass.
# ==========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# ==========================================================
# SECTION 6: LOAD THE DATASET
# ==========================================================
# Load the training and testing datasets directly from
# their respective folders.
#
# image_dataset_from_directory() automatically:
# • Reads all images.
# • Assigns labels based on folder names.
# • Resizes images.
# • Creates batches of images.
# ==========================================================

train_ds = image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
# ==========================================================
# SECTION 7: DISPLAY CLASS INFORMATION
# ==========================================================
# Retrieve the class names detected from the dataset and
# display useful information.
# ==========================================================

class_names = train_ds.class_names

print("Class Names :", class_names)
print("Number of Classes :", len(class_names))

# ==========================================================
# SECTION 8: VISUALIZE SAMPLE MRI IMAGES
# ==========================================================
# Display a few sample MRI images from the training dataset
# along with their corresponding class labels.
#
# This helps us verify:
# • Images are loaded correctly.
# • Labels are assigned correctly.
# • Image quality before training.
# ==========================================================

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.show()
# ==========================================================
# SECTION 9: CHECK DATASET SHAPE
# ==========================================================
# Display the shape of one batch of images and labels.
#
# Expected Output:
#
# Images Shape : (32, 224, 224, 3)
# Labels Shape : (32,)
#
# Meaning:
# 32  -> Batch Size
# 224 -> Image Height
# 224 -> Image Width
# 3   -> RGB Channels
# ==========================================================

for images, labels in train_ds.take(1):

    print("Images Shape :", images.shape)
    print("Labels Shape :", labels.shape)

# ==========================================================
# SECTION 10: DISPLAY NUMBER OF IMAGES IN EACH CLASS
# ==========================================================
# Count the number of training images available in each
# category. This helps identify whether the dataset is
# balanced or imbalanced.
# ==========================================================

print("Number of Images in Each Class\n")

for folder in os.listdir(train_path):

    folder_path = os.path.join(train_path, folder)

    image_count = len(os.listdir(folder_path))

    print(f"{folder:<12} : {image_count} images")
# ==========================================================
# SECTION 11: DATA PREPROCESSING (NORMALIZATION)
# ==========================================================
# Deep learning models perform better when input pixel
# values are within a small range.
#
# Original Pixel Range:
# 0 to 255
#
# After Normalization:
# 0.0 to 1.0
#
# The Rescaling layer divides every pixel value by 255,
# making the training process faster and more stable.
# ==========================================================

# Create a normalization layer
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Apply normalization to the training dataset
train_ds = train_ds.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Apply normalization to the testing dataset
test_ds = test_ds.map(
    lambda images, labels: (normalization_layer(images), labels)
)
# ==========================================================
# SECTION 12: VERIFY NORMALIZATION
# ==========================================================
# Check whether the pixel values have been normalized.
#
# Expected Output:
# Minimum Pixel Value : 0.0
# Maximum Pixel Value : 1.0
# ==========================================================

for images, labels in train_ds.take(1):

    print("Minimum Pixel Value :", images[0].numpy().min())
    print("Maximum Pixel Value :", images[0].numpy().max())
# ==========================================================
# SECTION 13: OPTIMIZE THE DATASET PIPELINE
# ==========================================================
# TensorFlow provides performance optimization techniques.
#
# cache():
# Stores the dataset in memory after the first epoch,
# reducing disk reads.
#
# prefetch():
# Loads the next batch while the current batch is being
# processed by the GPU/CPU, reducing idle time.
#
# These optimizations significantly improve training speed.
# ==========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
# ==========================================================
# SECTION 14: DATA AUGMENTATION
# ==========================================================
# Data Augmentation creates modified versions of existing
# images during training.
#
# Benefits:
# • Reduces overfitting
# • Improves model generalization
# • Makes the model more robust to image variations
#
# Applied Transformations:
# • Random Horizontal Flip
# • Random Rotation
# • Random Zoom
# ==========================================================

data_augmentation = tf.keras.Sequential([

    # Randomly flip images horizontally
    tf.keras.layers.RandomFlip("horizontal"),

    # Randomly rotate images by ±10%
    tf.keras.layers.RandomRotation(0.1),

    # Randomly zoom images by ±10%
    tf.keras.layers.RandomZoom(0.1)

])
# ==========================================================
# SECTION 15: VISUALIZE AUGMENTED IMAGES
# ==========================================================
# Display multiple augmented versions of the same MRI image.
#
# This helps verify that augmentation is working correctly
# before training the model.
# ==========================================================

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    first_image = images[0]

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0),
            training=True
        )

        plt.imshow(augmented_image[0])

        plt.axis("off")

plt.show()
normalization_layer = tf.keras.layers.Rescaling(1./255)
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(255.0),
    base_model,
    ...
])
# ==========================================================
# SECTION 16: LOAD THE PRE-TRAINED EFFICIENTNETB0 MODEL
# ==========================================================
# EfficientNetB0 is a pre-trained Convolutional Neural
# Network (CNN) trained on the ImageNet dataset containing
# over one million images.
#
# Instead of training a CNN from scratch, we use Transfer
# Learning to leverage the knowledge learned from ImageNet.
#
# Advantages:
# • Faster training
# • Better accuracy
# • Requires less data
# • Reduces overfitting
# ==========================================================

# Import EfficientNetB0 model
from tensorflow.keras.applications import EfficientNetB0

# Import Keras layers and model class
from tensorflow.keras import layers, models

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(
    weights="imagenet",        # Load ImageNet pre-trained weights
    include_top=False,         # Remove original classification layer
    input_shape=(224, 224, 3)  # Input image size
)

# Freeze all layers so only the custom classifier is trained
base_model.trainable = False
# ==========================================================
# SECTION 17: BUILD THE COMPLETE CLASSIFICATION MODEL
# ==========================================================
# We build a Sequential model by combining:
#
# 1. Data Augmentation
# 2. Rescaling Layer
# 3. EfficientNetB0 Feature Extractor
# 4. Global Average Pooling
# 5. Dropout Layers
# 6. Dense Hidden Layer
# 7. Output Layer
#
# The output layer contains 4 neurons because our dataset
# has four classes.
# ==========================================================

model = models.Sequential([

    # Apply random transformations during training
    data_augmentation,

    # Convert pixel values back to EfficientNet's expected range
    # (Kept to match the current trained model.)
    layers.Rescaling(255.0),

    # Pre-trained EfficientNetB0 feature extractor
    base_model,

    # Convert feature maps into a single feature vector
    layers.GlobalAveragePooling2D(),

    # Reduce overfitting by randomly dropping neurons
    layers.Dropout(0.3),

    # Hidden fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Additional dropout layer
    layers.Dropout(0.2),

    # Output layer
    # Softmax returns probability for each class
    layers.Dense(
        4,
        activation="softmax"
    )
])
# ==========================================================
# SECTION 18: DISPLAY MODEL SUMMARY
# ==========================================================
# Display the architecture of the complete neural network.
#
# The summary includes:
# • Layer names
# • Output shapes
# • Number of parameters
# • Trainable parameters
# • Non-trainable parameters
# ==========================================================

model.summary()
# ==========================================================
# SECTION 19: VERIFY MODEL INPUT AND OUTPUT SHAPES
# ==========================================================
# Verify that the model accepts images of size
# 224 × 224 × 3 and produces predictions for
# four output classes.
# ==========================================================

print("Input Shape :", model.input_shape)
print("Output Shape :", model.output_shape)
# ==========================================================
# SECTION 20: DISPLAY TRAINABLE VARIABLES
# ==========================================================
# Since the EfficientNetB0 model is frozen,
# only the custom classification layers are trainable.
#
# This significantly reduces training time.
# ==========================================================

print("Number of Trainable Variables :", len(model.trainable_variables))
# ==========================================================
# SECTION 21: COMPILE THE MODEL
# ==========================================================
# Before training, the model must be compiled.
#
# Optimizer:
# Adam optimizer is used because it provides faster and
# more stable convergence.
#
# Loss Function:
# Sparse Categorical Crossentropy is used because the
# labels are integer encoded (0, 1, 2, 3).
#
# Metric:
# Accuracy is used to evaluate the model's performance.
# ==========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# ==========================================================
# SECTION 22: CONFIGURE TRAINING CALLBACKS
# ==========================================================
# Callbacks help improve the training process.
#
# EarlyStopping:
# Stops training if validation loss does not improve,
# helping prevent overfitting.
#
# ModelCheckpoint:
# Saves the model with the best validation accuracy.
# ==========================================================

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save the best performing model
checkpoint = ModelCheckpoint(
    "brain_tumor_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)
# ==========================================================
# SECTION 23: TRAIN THE MODEL
# ==========================================================
# Train the model using the training dataset.
#
# Validation data is used after every epoch to evaluate
# how well the model performs on unseen images.
#
# Epoch:
# One complete pass through the entire training dataset.
# ==========================================================

# Number of training epochs
EPOCHS = 15

# Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)
# ==========================================================
# SECTION 24: DISPLAY TRAINING HISTORY
# ==========================================================
# Display all metrics recorded during training.
#
# These values are later used to plot graphs.
# ==========================================================

print(history.history.keys())
# ==========================================================
# SECTION 25: PLOT TRAINING AND VALIDATION ACCURACY
# ==========================================================
# Plot training accuracy and validation accuracy
# to observe how the model improves over epochs.
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()
# ==========================================================
# SECTION 26: PLOT TRAINING AND VALIDATION LOSS
# ==========================================================
# Plot training loss and validation loss.
#
# A decreasing loss indicates that the model is learning.
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()
# ==========================================================
# SECTION 27: EVALUATE THE TRAINED MODEL
# ==========================================================
# Evaluate the trained model using the testing dataset.
#
# The evaluation returns:
# • Test Loss
# • Test Accuracy
# ==========================================================

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
# ==========================================================
# SECTION 28: GENERATE PREDICTIONS FOR THE TEST DATASET
# ==========================================================
# Predict the class labels for every image in the testing
# dataset. These predictions will be used to generate the
# confusion matrix and classification report.
# ==========================================================

# Import evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report

# Import Seaborn for visualization
import seaborn as sns

# Lists to store actual and predicted labels
y_true = []
y_pred = []

# Predict each batch of test images
for images, labels in test_ds:

    # Get prediction probabilities
    predictions = model.predict(images, verbose=0)

    # Store actual labels
    y_true.extend(labels.numpy())

    # Store predicted labels
    y_pred.extend(np.argmax(predictions, axis=1))
# ==========================================================
# SECTION 29: CONFUSION MATRIX
# ==========================================================
# A confusion matrix compares the actual class labels with
# the predicted class labels.
#
# Diagonal values:
# Correct predictions
#
# Off-diagonal values:
# Incorrect predictions
# ==========================================================

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.show()
# ==========================================================
# SECTION 30: CLASSIFICATION REPORT
# ==========================================================
# Display important evaluation metrics for each class.
#
# Metrics:
# Precision
# Recall
# F1-Score
# Support
# ==========================================================

print("Classification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
# ==========================================================
# SECTION 31: SAVE THE TRAINED MODEL
# ==========================================================
# Save the trained model in Keras format.
#
# This allows us to reuse the model later without
# retraining it.
# ==========================================================

model.save("brain_tumor_model.keras")

print("Model saved successfully.")
# ==========================================================
# SECTION 32: ENABLE FINE-TUNING
# ==========================================================
# Fine-tuning improves model performance by allowing the
# last few layers of EfficientNetB0 to learn features
# specific to brain MRI images.
#
# Instead of training the entire network, only the last
# 20 layers are unfrozen.
# ==========================================================

# Unfreeze the EfficientNet model
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

print("Trainable Layers:\n")

for layer in base_model.layers[-20:]:
    print(layer.name, "-", layer.trainable)
# ==========================================================
# SECTION 33: COMPILE THE MODEL FOR FINE-TUNING
# ==========================================================
# Use a very small learning rate to avoid making large
# changes to the pre-trained weights.
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
# ==========================================================
# SECTION 34: REDUCE LEARNING RATE ON PLATEAU
# ==========================================================
# If the validation loss stops improving, reduce the
# learning rate automatically.
#
# This helps the model converge more smoothly.
# ==========================================================

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-7

)
# ==========================================================
# SECTION 35: FINE-TUNE THE MODEL
# ==========================================================
# Continue training using the partially unfrozen
# EfficientNetB0 model.
#
# Fine-tuning allows the network to learn features that
# are more relevant to brain MRI classification.
# ==========================================================

fine_tune_epochs = 10

history_fine = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=fine_tune_epochs,

    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]

)
# ==========================================================
# SECTION 36: EVALUATE THE FINE-TUNED MODEL
# ==========================================================
# Evaluate the performance of the fine-tuned model on the
# testing dataset.
# ==========================================================

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Fine-Tuned Test Loss : {test_loss:.4f}")
print(f"Fine-Tuned Test Accuracy : {test_accuracy:.4f}")
# ==========================================================
# SECTION 37: PLOT FINE-TUNING ACCURACY
# ==========================================================
# Plot the training and validation accuracy obtained during
# the fine-tuning phase.
#
# This graph helps us visualize whether the model's
# performance improved after unfreezing the last layers of
# EfficientNetB0.
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history_fine.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_fine.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Fine-Tuning Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================================
# SECTION 38: PLOT FINE-TUNING LOSS
# ==========================================================
# Plot the training and validation loss during
# fine-tuning.
#
# Lower validation loss generally indicates better
# generalization on unseen data.
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history_fine.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_fine.history["val_loss"],
    label="Validation Loss"
)

plt.title("Fine-Tuning Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================================
# SECTION 38: PLOT FINE-TUNING LOSS
# ==========================================================
# Plot the training and validation loss during
# fine-tuning.
#
# Lower validation loss generally indicates better
# generalization on unseen data.
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history_fine.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_fine.history["val_loss"],
    label="Validation Loss"
)

plt.title("Fine-Tuning Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================================
# SECTION 40: LOAD THE SAVED MODEL
# ==========================================================
# Load the saved model from Google Drive.
#
# This demonstrates that the model can be restored and
# used later for prediction.
# ==========================================================

model = load_model(
    "/content/drive/MyDrive/BrainTumorProject/brain_tumor_model.keras"
)

print("Model loaded successfully.")
# ==========================================================
# SECTION 41: UPLOAD A NEW MRI IMAGE
# ==========================================================
# Upload an MRI image from your computer for prediction.
#
# Google Colab will display a file upload dialog.
# ==========================================================

from google.colab import files

uploaded = files.upload()
# ==========================================================
# SECTION 42: LOAD AND DISPLAY THE MRI IMAGE
# ==========================================================
# Load the uploaded MRI image, resize it to 224 × 224,
# and display it.
#
# This ensures the image matches the input size expected
# by EfficientNetB0.
# ==========================================================

# Replace with the uploaded filename if necessary
img_path = "Te-aug-me_2.jpg"

# Load image
img = image.load_img(
    img_path,
    target_size=(224,224)
)

# Display image
plt.figure(figsize=(5,5))

plt.imshow(img)

plt.title("Input MRI Image")

plt.axis("off")

plt.show()
# ==========================================================
# SECTION 43: PREPROCESS THE MRI IMAGE
# ==========================================================
# Convert the image into a NumPy array, add a batch
# dimension, and normalize pixel values.
#
# The model expects input in the form:
#
# (Batch Size, Height, Width, Channels)
#
# Example:
# (1, 224, 224, 3)
# ==========================================================

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize pixel values
img_array = img_array / 255.0
# ==========================================================
# SECTION 44: PREDICT THE MRI IMAGE
# ==========================================================
# Predict the class of the uploaded MRI image.
#
# The model outputs the probability of each class.
#
# The class with the highest probability is selected as
# the final prediction.
# ==========================================================

# Predict probabilities
prediction = model.predict(img_array)

# Display probability for each class
print("Prediction Probabilities:\n")

print(prediction)

# Get predicted class index
predicted_class = np.argmax(prediction)

# Get confidence score
confidence = np.max(prediction) * 100

# Display prediction
print("\nPredicted Class :", class_names[predicted_class])

print(f"Confidence : {confidence:.2f}%")
