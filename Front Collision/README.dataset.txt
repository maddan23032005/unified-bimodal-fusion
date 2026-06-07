# Self Driving Car Re-Encode > raw6
https://universe.roboflow.com/brad-dwyer/self-driving-car-re-encode

Provided by [Roboflow](https://roboflow.ai)
License: Public Domain

# Overview

The original Udacity Self Driving Car Dataset is missing labels for thousands of pedestrians, bikers, cars, and traffic lights. This will result in poor model performance. When used in the context of self driving cars, this could even lead to human fatalities.

We re-labeled the dataset to correct errors and omissions. We have provided convenient downloads in many formats including VOC XML, COCO JSON, Tensorflow Object Detection TFRecords, and more.

Some examples of labels missing from the original dataset:
![Examples of Missing Labels](https://i.imgur.com/A5J3qSt.jpg)

# Use Cases

Udacity is building an open source self driving car! You might also try using this dataset to do person-detection and tracking.

# Using this Dataset

Our updates to the dataset are released under the same license as the original.

**Note:** the dataset contains many duplicated bounding boxes for the same subject which we have not corrected. You will probably want to filter them by taking the IOU for classes that are 100% overlapping or it could affect your model performance (expecially in stoplight detection which seems to suffer from an especially severe case of duplicated bounding boxes).

# About Roboflow

[Roboflow](https://roboflow.ai) makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless.

Developers reduce 50% of their boilerplate code when using Roboflow's workflow, save training time, and increase model reproducibility.
:fa-spacer:
#### [![Roboflow Wordmark](https://i.imgur.com/WHFqYSJ.png =350x)](https://roboflow.ai)
