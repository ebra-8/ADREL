Implementation of Adversarial Representation Learning

Requirements: Keras, TensorFlow, Python

This document summarizes the steps to obtain the adversarial representations via ADREL in Keras with TensorFlow backend.

As an example, the following steps can be performed to generate representations for Russian language.

**Notes:** 

* If you want to  run the model on GloVe wword embeddings you need to run GloVe_embeddings.py first (before Steps 1 and 2). Since the word embeddings are large (a few gigabytes), we did not include them in this repository. The **GloVe_embeddings** folder contains the results of such training for English, French, and Italian. The Russian Embeddings are large (2.6 GB), so we did not include them in the folder. If you need access to them, please contact us, and we will be happy to provide a Google Drive link. 

* Please change the paths to the data folder.

**Step 1)** Run LSTM_Russian.py to get the input sequence for the target language. The same code needs to be run on English to obtain the source language sequences.

**Step 2)** Run Invariant_representations.py to obtain the emitted representations of the input from BiLSTM, based on which adversarial training is conducted to obtain the domain-invariant representation and then classification result on the projected data is obtained via an LSTM classifier.
 
 The code has been tested on GPU. However, it should be runnable on CPU as well.
 
 Running the baselines:
 
 The benchmark folder contains the baselines as well the benchmark methods that can be used to compare to ADREL.
 CNN_Kim2014_TF has the code for CNN-sentence

**ADREL Model Specification**
ADREL was implemented using Keras Python package on a single Linux machine with a GeForce GTX 1060 Graphical Processing Unit (GPU) with 6 GB of memory and 1,280 Cuda cores. To further ensure the reproducibility of our proposed model, here we report the exact architectural details as well as the parameter or hyper-parameter settings. The initial word embeddings for the first phase of ADREL were obtained by GloVe word representations (Pennington et al. 2015) applied to cybersecurity documents from our data collection. We included the initial word representations in the provided repository. For the second stage of ADREL, logistic (cross-entropy) loss was used to update model parameters, given its success in prior text classification applications (Goldberg 2017). To alleviate the need to tune learning rates (Goldberg 2017), Adam optimizer (Kingma and Ba 2015) was used to learn the model’s parameters. The specifications of the proposed model is given below. 

| Component | Layer  | # of Units | Activation Type | Output Shape         | Layer-Specific Parameters     |
| :-----:   | :-:    | :-:        | :-:             | :-:                  |     :-:                       |
| Generator | Embedding | 64      | -               | (Batch_size,150, 64) |     Vocabulary_ size = 10,000 |

We have designed the model parsimoniously. That is, the architecture of the generator and discriminator are intended to be simple and to have few parameters. As for activation functions in BiLSTM, hyperbolic tangent (tanh) was used due to its proven performance in text analytics (Goldberg 2017).

