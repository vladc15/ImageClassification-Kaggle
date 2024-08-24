# Image Classification - Kaggle Competition

## Project Overview

This project aims to classify images using Convolutional Neural Networks (CNNs) for a Kaggle competition. The dataset consists of images divided into three classes, with a training set, validation set, and test set provided. The project involves the implementation of a deep learning model with various computer vision techniques to enhance performance and ensure robust classification.

There were 2 approaches used in this competition:
- simple SVMs - using one-vs-one strategy, with an accuracy of 34%
- Convolutional Neural Network (CNN), with an accuracy of 76%

## Dataset

The dataset consists of 10,500 training images, each 80x80 pixels, with three color channels (RGB). There are 3,000 images in the validation set and 4,500 images in the test set. The data was preprocessed by normalizing each color channel to ensure consistency in the model's input.

## Model Architecture

The model was built using a CNN architecture designed to extract features from the images and perform classification. The key components of the architecture are:

1. **Convolutional Layers**: 
   - 6 convolutional layers were used to capture spatial hierarchies in the images.
   - Each convolutional layer is followed by a ReLU activation function and batch normalization to stabilize and accelerate training.

2. **Pooling Layers**: 
   - Max Pooling layers were used after certain convolutional layers to downsample the feature maps, reducing the spatial dimensions and keeping the most prominent features.

3. **Fully Connected Layers**: 
   - 3 fully connected layers are used to transform the feature maps into a classification decision. 

4. **Dropout**: 
   - A dropout rate of 0.5 was applied to prevent overfitting by randomly setting a fraction of the input units to zero during training.

## Training Details

- **Optimizer**: Adam optimizer with an initial learning rate of 0.001.
- **Learning Rate Scheduler**: A learning rate scheduler was implemented to reduce the learning rate when a plateau in validation accuracy was detected, with a `patience` parameter set to 3.
- **Loss Function**: Cross-entropy loss function, suitable for multi-class classification problems.
- **Batch Size**: 32.
- **Dropout**: 0.5 rate in fully connected layers to prevent overfitting.
- **Epochs**: The model was trained for 60 epochs.

## Results and Evaluation

The performance of the model was monitored using the validation set, and the final evaluation was conducted on the test set. The model's performance metrics, such as accuracy and loss, were used to determine the model's effectiveness in classifying the images correctly.
