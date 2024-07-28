# Dota 2 Match Predictor Documentation

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
\[ X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} \]

- $ X' $ is the scaled value.
- $ X $ is the original value.
- $ X_{\text{min}} $ is the minimum value of the feature.
- $ X_{\text{max}} $ is the maximum value of the feature.
\
\
This is really useful here, because the features are numerical, but also because they have different scales and you want to bring them to a common scale, without distorting the differences in the ranges of values.\
\
Scaling helps ensure uniformity in gradient descent across all features, leading to faster and more reliable convergence while avoiding suboptimal solutions. This technique actually increased the model's **accuracy** to **58-59%**.

2. [**Synthetic Minority Over-sampling Technique (SMOTE)**](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)\
A technique used to address class imbalance in datasets, particularly useful to every classification problem.\
For each instance in the minority class, SMOTE finds its k-nearest neighbors (we used k=5). It then randomly selects one of these neighbors and *"a synthetic example is created at a randomly selected point between the two examples in feature space"*, using the formula:
$$ X_{\text{new}} = X_{\text{instance}} + \lambda \times (X_{\text{neighbor}} - X_{\text{instance}}) $$

- $ X_{\text{new}} $ is the synthetic instance.
- $ X_{\text{instance}} $ is the original minority class instance.
- $ X_{\text{neighbor}} $ is the selected neighbor.
- $ \lambda $ is a random number between 0 and 1. \
\
This process augmented the original dataset by approximately **4,000 instances**, increasing the overall classification performance. After preprocessing and final augmentation, we achieved a **73/17/10 split** for training, validation, and testing, respectively.

