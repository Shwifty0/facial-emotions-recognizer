# Facial Emotions Recognizer

## Description

A streamlit app integrated with a deep-learning model that recognizes facial emotions in images.

## Usage

### Installing

In order to use this makefile you will need to make sure that the following
dependencies are installed on your system:
  - Pillow==9.3.0
  - streamlit==1.26.0
  - torch==2.0.1
  - torchvision==0.15.2


### Setup generic data

```yml
---
title: Facial Emotions Recognizer
author: Muhammad Ozair
language: en-US
---
```


# Introduction

This is a Facial Emotion recognizer Streamlit app that takes your selfie as an input and predicts your emotions based on these categories:
['Angry', 'Contempt', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

## Model Building Process
- Initially the idea was to implement and recreate the Res-net 18 Architecture from scratch and use it for training to gain a deep understanding of how the model works
- Eventually, I ended up importing the model architecture with pre-loaded weights, using PyTorch's built-in model library.
- For the Dataset, I used [Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition) from Kaggle



## Streamlit and Deployment of the Deep Learning Model.

- I used Streamlit for building a web-application for real-time use of the model, which was new learning experience for me.
- Streamlit module has ton's of useful front-end functionalities that let you realise your Machine Learning / Deep Learning models within a matter of hours rather than building web-apps traditionally that can take days.
