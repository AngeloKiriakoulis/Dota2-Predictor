# Dota 2 Match Predictor Documentation

## Outline

* [Introduction](#introduction)
* [Objective](#objective)
* [Data Collection and Preparation](#data-collection-and-preparation)
* [Advanced Data Preprocessing](#advanced-data-preprocessing)
* [Model Architecture](#model-architecture)
* [Model Training](#model-training)
* [Evaluation](#evaluation)

## Introduction

The notebook we built aims to predict the outcome of Dota 2 matches using a custom deep learning model. The dataset comprises various features related to the matches, and the goal is to develop a model that can forecast the winner as accurately as some benchmark classifiers.

In this Markdown file, we will explain what tools we used, the steps we took for data preparation, and the design of our custom deep learning model so we can compare it to famous benchmark classifiers.

## Objective

The objective of this project is to develop a predictive model for Dota 2 match outcomes using deep learning techniques. By analyzing historical match data, we aim to decide which team will win, based on various in-game features.

## Data Collection and Preparation

We downloaded the Dota 2 Datasets from the [UC Irvine ML Repository](https://archive.ics.uci.edu/dataset/367/dota2+games+results), that contains 102944 instances. Each instance represents various game characteristics, such as the winning team, game type, and champion selection. I chose not to delve into feature engineering to show the remarkable capability of a simple deep learning model in discovering hidden patterns.

The train and test datasets were already split by the repository authors. We decided to take it a step further by creating a **validation dataset**, which will help us evaluate the model's accuracy in generalizing. As the final step in our initial data preprocessing, we separated the datasets into input features and the target feature, preparing them for model training, and named each feature.

## Advanced Data Preprocessing

During model training, following the simple preprocessing, the accuracy achieved was not exceeding **52-53%**, which is no better than random guessing. I used two advanced techniques to improve the model's accuracy to more reasonable levels:

1. [**MinMax Scaling**](https://medium.com/@poojaviveksingh/all-about-min-max-scaling-c7da4e0044c5) \
A normalization technique used to scale the features of your data to a fixed range, typically [0, 1] or [-1, 1]. \
It works by simply applying the scaling formula to each value in the feature:

$$X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

* $X'$ is the scaled value.
* $X$ is the original value.
* $X_{\text{min}}$ is the minimum value of the feature.
* $X_{\text{max}}$ is the maximum value of the feature.
\
\
This is really useful here, because the features are numerical, but also because they have different scales and you want to bring them to a common scale, without distorting the differences in the ranges of values.\
\
Scaling helps ensure uniformity in gradient descent across all features, leading to faster and more reliable convergence while avoiding suboptimal solutions. This technique actually increased the model's **accuracy** to **58-59%**.

2. [**Synthetic Minority Over-sampling Technique (SMOTE)**](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)\
A technique used to address class imbalance in datasets, particularly useful to every classification problem.\
For each instance in the minority class, SMOTE finds its k-nearest neighbors (we used k=5). It then randomly selects one of these neighbors and *"a synthetic example is created at a randomly selected point between the two examples in feature space"*, using the formula:

$$X_{\text{new}} = X_{\text{instance}} + \lambda \times (X_{\text{neighbor}} - X_{\text{instance}})$$

* $X_{\text{new}}$ is the synthetic instance.
* $X_{\text{instance}}$ is the original minority class instance.
* $X_{\text{neighbor}}$ is the selected neighbor.
* $\lambda$ is a random number between 0 and 1. \
\
This process augmented the original dataset by approximately **4,000 instances**, increasing the overall classification performance. After preprocessing and final augmentation, we achieved a **73/17/10 split** for training, validation, and testing, respectively.

## Model Architecture

To build, deploy and evaluate our Deep Neural Network we used the [TesnorFlow Keras Framework](https://keras.io/), to predict the outcome of Dota 2 matches.

It consists of three fully connected hidden layers, activating each output by the ReLU activation function. Each layer is followed by a dropout layer to prevent overfitting. Dropout layers drop a certain percentage (20% in our case) of neurons randomly, forcing the network to learn more robust and generalized features.

The final layer contains a single neuron with sigmoid activation, which outputs a predicted probability that a particular Dota 2 match will result in a win for the team being analyzed (team that is of class=1). These probabilities are between 0 and 1, where a value closer to 1 indicates a higher probability of winning.

## Model Training

The model is compiled with the **Adam optimizer**, binary cross-entropy loss function, and accuracy as the evaluation metric. The **binary cross-entropy loss** function is appropriate for our binary classification problem, where the goal is to predict the probability of a binary outcome (win or lose).

A tweak to the optimizer's **learning rate** results in variations of smoothness in the model's accuracy. I found out that a learning rate of 0.0001 strikes the right balance of smoothness and convergence speed.

To prevent overfitting and improve the model's generalization, we use early stopping. **Early stopping** monitors the validation loss and stops training when the loss has not improved for a specified number of epochs (patience). We set the patience to 5 epochs and enabled restoring the best weights observed during training.

We trained the model using the training dataset and validated it with the validation dataset. The training process runs with a **batch size of 128**, which means that the network will compute the error for 128 instances, before updating its weights.

We used a max limit of 50 epochs for our model. But as we analyzed, the early stopping callback will be used to halt training if the validation loss does not improve for 5 consecutive epochs, thereby preventing overfitting.

## Evaluation

We put our model up against two classic classifiers, to benchmark our model:

* [Logistic Regressor](https://en.wikipedia.org/wiki/Logistic_regression), Validation Accuracy:*~59,6%* Test Accuracy:*~59,6%*
* [Random Forest Classifier](https://en.wikipedia.org/wiki/Random_forest),Validation Accuracy:*~57,6%* Test Accuracy:*~57,9%* 

Our model obtained a Valuation Accuracy of **59.61%** and sometimes even higher and a Test Accuracy of **59.1%**.

While our model lacks preprocessing or advanced techniques, it provides an improvement over some of the benchmark models. Keep in mind that further enhancements (such as feature engineering, hyperparameter tuning, or more complex architectures) could potentially yield even better results.


