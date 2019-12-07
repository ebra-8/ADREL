Implementation of Adversarial Representation Learning

Requirements: Keras, TensorFlow, Python

This document summarizes the steps to obtain the adversarial representations via ADREL in Keras.

As an example, the following steps can be performed to generate representations for Russian language.

Step 1) Run PrepareRussian.py to get the input sequence for the target language. The same code needs to be run on English to obtain the source language sequences. 

Step 2) Run dnm_Adv_EnRu_NoMT_2.py to obtain the emitted representations of the input from BiLSTM. The BiLSTM representations will be saved on the disk in HDF format
 under output_adv_NoMT_ru/Ru_states_2.hdf5.

Step 3) Input the obtained BiLSTM representations to GAN_v2_FinalClassificationwithLSTm_RU_400_2. This would run adversarial training for 400 iterations to obtain the domain-invariant representation, saves the trained generator in another HDF file named gen2_saved_ru_4k_2.h5, and applies it on the training and testing set for Russian  and save them under output_adv_NoMT_ru/projected_rep_Ru_train_4k_2.hdf5 and output_adv_NoMT_ru/projected_rep_Ru_test_4k_2.hdf5, respectively. 

Step 4) Run ApplyGAN_RU_400_2.py to obtain the classification result on the projected data via a BiLSTM classifier. The output for each iteration will be saved as
 AUC_Ru_GAN/Run_4k_2_x, in which x is the number of training iterations
 
 
 Running the baselines:
 
 The benchmark folder contains the baselines as well the benchmark methods that can be used to compare to ADREL.
 CNN_Kim2014_TF has the code for CNN-sentence

* Please change the paths to the data folder.



