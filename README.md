# MNIST-Dataset

## Introduction

This analysis was part of a graduate level machine learning course at Kent State University. The main goal of the project was to analyse the MNIST dataset and perform PCA and visualize the different principal components and the amount of information they each contain.
Additionally, we trained a neural network that was used to classify the hand-written images into digits ranging from 0-9.

## Tools and methods used

The tools that were used during this project are:
1. Python libraries such as Tensorflow, Keras, Pandas, Matplotlib...
2. Principle component analysis
3. Convolutional Neural Networks

## Findings

By the end of the project we were able to boil down the number of principle components into 10, which were used to classify the handwritten images in the MNIST dataset. We also visualized the amount of information stored in each principle component which led us to choose 10 as an optimal number. Furthermore, we calculated the mean of a random image from the dataset and visualized it. Finally, we visualized digits when they were reduced to 1 up to 10 principle components to highlight the differences.
Additionally, we trained the convolutional neural network on the training dataset using 100 epochs and a batch size of 500, which yielded an accuracy of 94% on the training data and 92% on the test data. These figures can be further improved by using a higher number of epochs and changing the batch size.
