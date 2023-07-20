# Machine Learning Research Project (1).
 Credits to Deep Learning basics: Introduction and Overview
 by Lex Fridman.
 
 To watch full video: https://youtu.be/O5xeyoRL95U

# Results
 This project outputs an arrucacy around 97% on the test data.

# Summary
This Python program is an image recognition model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The code loads the dataset, normalizes the pixel values to a range of 0 to 1, and creates a simple neural network model with two dense layers. The first dense layer has 128 neurons with ReLU activation, and the second dense layer has 10 neurons with a softmax activation for multi-class classification.

The model is then compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the metric. It is trained on the training images and labels for 5 epochs. After training, the model is evaluated on the test dataset, and the test accuracy is printed, which is approximately 97%.

Finally, the model is used to make predictions on the test images, and the predictions are stored in the predictions variable. This program demonstrates a basic image recognition model using TensorFlow and Keras for digit recognition, achieving an accuracy of around 97% on the test data.

## Purpose
This short snippet of code is used for my own learning journey and I plan to publish most all content related to my machine learning journey.

## Notes
MNIST Dataset: The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. 
Keras: Keras is an open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
ReLu: ReLU activation function is an activation function defined as the positive part of its argument: where x is the input to a neuron.
Sparse categorical cross-entropy: An extension of the categorical cross-entropy loss function that is used when the output labels are represented in a sparse matrix format. In a sparse matrix format, the labels are represented as a single index value rather than a one-hot encoded vector.
Epoch: The one entire passing of training data through the algorithm.
