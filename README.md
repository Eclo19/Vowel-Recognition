# Vowel Sound Classification Using K-Nearest Neighbors (KNN) Classifier

## Overview:
This project is an introductory exploration into machine learning, specifically focusing on audio classification using the K-Nearest Neighbors (KNN) algorithm. The primary goal of this project is to classify vowel sounds extracted from WAV audio files. The main feature used for classification is the ratio of energy across different frequency bands, a global feature extracted from each audio file.

## Training Data:
I recorded myself producing five vowel sounds five times. The recordings were done in Logic Pro with an SM57 microphone, and they all have a sample rate of 44100 Hz and a bit depth of 16. The vowels are: 
  
  -AAA (as in fAther)
  
  -EEE (as in mEt)
  
  -III (as in wInd)
  
  -OOO (as in lOt)
  
  -UUU (as in mOOd)

## Features:
Energy Bands Ratios
The sole feature extracted from the audio data in this project is the energy bands ratios. The energy in each frequency band of the audio signal is compared to the energy in the first band, and these ratios are packed into a feature vector. This feature captures the distribution of energy across the frequency spectrum, which is crucial for distinguishing between different vowel sounds.

## How the Code Works:
* The **train()** function is responsible for training the KNN classifier. The function reads each audio file in the training dataset, normalizes it, and extracts the energy bands ratios using the energy_per_frequency_band() helper function. Each feature vector is associated with its corresponding label (vowel sound). The labeled feature vectors are stored in a dictionary (self.training_features), where the keys are labels, and the values are lists of feature vectors.
* The **predict()** function predicts the label of a new audio sample based on the KNN algorithm. The input audio data is first normalized, then the same feature extraction process used in the **train()** function is applied to the input data to create a feature vector. The Euclidean distances between the input feature vector and all feature vectors in the training dataset are calculated using the **euclidean_distances()** helper function. The distances are sorted, and the k nearest neighbors (i.e., the feature vectors with the smallest distances) are selected. The labels of the nearest neighbors are determined, and the most frequent label is chosen as the predicted label. In case of a tie, the label associated with the smallest distance (first in the sorted list) is selected.

## Helper Functions:

  * **energy_per_frequency_band()**: This function computes the energy in different frequency bands of the audio signal and calculates the ratios relative to the first band.
  
  * **normalize()**: This function normalizes the input audio data, preparing it for feature extraction.
  
  * **euclidean_distances()**: This helper function computes the Euclidean distances between a given feature vector and the feature vectors in the training dataset.
