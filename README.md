# Potato-Disease-Detection-Using-Deep-Learning-Approaches


This project aims to develop a deep learning model for detecting and classifying diseases in potato plants. The project is implemented using Python and TensorFlow.
Dataset

The dataset used for this project contains images of healthy potato plants and plants infected with four common diseases: early blight, late blight, leaf curl, and mosaic virus. The dataset is divided into training, validation, and test sets.
Model Architecture

Two deep learning models were developed for this project: a convolutional neural network (CNN) and a transfer learning model using the VGG16 architecture.
CNN Model

The CNN model consists of multiple convolutional and pooling layers followed by a fully connected output layer. The model is trained using the categorical cross-entropy loss function and the Adam optimizer.
Transfer Learning Model

The transfer learning model uses the VGG16 architecture pre-trained on the ImageNet dataset. The pre-trained layers are frozen, and a new fully connected output layer is added on top of the model. The model is fine-tuned on the potato disease dataset using the categorical cross-entropy loss function and the Adam optimizer.
Results

Both models were able to achieve high accuracy on the test set

    CNN Model: 97.20%
    Transfer Learning Model: 98.0%
