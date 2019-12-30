Implementation of Adversarial Representation Learning

Requirements: Keras, TensorFlow, Python

This document summarizes the steps to obtain the adversarial representations via ADREL in Keras.

As an example, the following steps can be performed to generate representations for Russian language.

Step 1) Run LSTM_Russian.py to get the input sequence for the target language. The same code needs to be run on English to obtain the source language sequences.

Step 2) Run Invariant_representations.py to obtain the emitted representations of the input from BiLSTM, based on which adversarial training is conducted to obtain the domain-invariant representation and then classification result on the projected data is obtained via an LSTM classifier.
 
 Running the baselines:
 
 The benchmark folder contains the baselines as well the benchmark methods that can be used to compare to ADREL.
 CNN_Kim2014_TF has the code for CNN-sentence

* Please change the paths to the data folder.



