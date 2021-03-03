**Implementation of Adversarial Representation Learning**

Requirements: Keras, TensorFlow, Python

This document summarizes the steps to obtain the adversarial representations via ADREL in Keras with TensorFlow backend.

As an example, the following steps can be performed to generate representations for Russian language.

**Notes:** 

* If you want to  run the model on GloVe wword embeddings you need to run GloVe_embeddings.py first (before Steps 1 and 2). Since the word embeddings are large (a few gigabytes), we did not include them in this repository. The **GloVe_embeddings** folder contains the results of such training for English, French, and Italian. The Russian Embeddings are large (2.6 GB), so we did not include them in the folder. If you need access to them, please contact us, and we will be happy to provide a Google Drive link. 

* Please change the paths to the data folder.

**Step 1)** Run LSTM_Russian.py to get the input sequence for the target language. The same code needs to be run on English to obtain the source language sequences.

**Step 2)** Run Invariant_representations.py to obtain the emitted representations of the input from BiLSTM, based on which adversarial training is conducted to obtain the domain-invariant representation and then classification result on the projected data is obtained via an LSTM classifier.
 
 The code has been tested on GPU. However, it should be runnable on CPU as well.
 
 **Running the Deep Learning-based Benchmarks**
 
 The benchmark folder contains the baselines as well the benchmark methods that can be used to compare to ADREL.
 CNN_Kim2014_TF has the code for CNN-sentence

**ADREL Model Specification**

ADREL was implemented using Keras Python package on a single Linux machine with a GeForce GTX 1060 Graphical Processing Unit (GPU) with 6 GB of memory and 1,280 Cuda cores. To further ensure the reproducibility of our proposed model, here we report the exact architectural details as well as the parameter or hyper-parameter settings. The initial word embeddings for the first phase of ADREL were obtained by GloVe word representations (Pennington et al. 2015) applied to cybersecurity documents from our data collection. We included the initial word representations in the provided repository. For the second stage of ADREL, logistic (cross-entropy) loss was used to update model parameters, given its success in prior text classification applications (Goldberg 2017). To alleviate the need to tune learning rates (Goldberg 2017), Adam optimizer (Kingma and Ba 2015) was used to learn the model’s parameters. The specifications of the proposed model is given below. 

| Component     | Layer              | # of Units | Activation Type | Output Shape         | Layer-Specific Parameters      |
| :-----:       | :-:                | :-:        | :-:             | :-:                  |     :-:                        |
| Generator     | Embedding          | 64         | -               | (Batch_size,150, 64) |     Vocabulary_ size = 10,000  |
| Generator     | Bidirectional LSTM | 128        | tanh            | (Batch_size,150, 256)|     Bias_initializer = 0       |
| Generator     | Bidirectional LSTM | 128        | tanh            | (Batch_size,150, 256)|     Bias_initializer = 0       |
| Discriminator | Bidirectional LSTM | 128        | tanh            | (Batch_size,150, 256)|     Bias_initializer = 0       |
| Discriminator | Bidirectional LSTM | 64         | tanh            | (Batch_size,64)      |     Bias_initializer = 0       |
| Discriminator | Dense              | 1          | sigmoid         | (Batch_size,1)       |     -                          |
| Classifier    | Input Layer        | -          | -               | (Batch_size,150, 256)|     Sparse = false             |
| Classifier    | Bidirectional LSTM | 128        | tanh            | (Batch_size, 256)    |     Bias_initializer = 0       |
| Classifier    | Dense              | 1          | sigmoid         | (Batch_size,1)       |     -                          |

We have designed the model parsimoniously. That is, the architecture of the generator and discriminator are intended to be simple and to have few parameters. As for activation functions in BiLSTM, hyperbolic tangent (tanh) was used due to its proven performance in text analytics (Goldberg 2017).

**Lexicon Construction for Keyword-based Baseline**

We designed a baseline experiment for a lexicography-based hacker asset detection method, which conducts a series of vocabulary searches to find related keywords from a lexicon within the textual content in the dark web platforms. To our knowledge, there is no multilingual lexicon specifically for hacker asset detection. Thus, we constructed a customized lexicon with 1,059 hacker asset identifiers by compiling and modifying five publicly-available lexicons as well as incorporating the indicators suggested by two subject matter experts after scanning product descriptions from dark web platforms. The compilation process is detailed in seven steps: 
**Step 1)** “Explore Terms,” a cybersecurity lexicon compiled by the Department of Homeland Security (DHS) (NICCS 2019). This lexicon was constructed as part of the National Initiative for Cybersecurity Careers and Studies (NICCS) program, and to our knowledge, is one of the most comprehensive lexicons in the field. This lexicon also complements other previous lexicons such as the NISTIR (National Institute of Standards and Technology Internal Reports). While this lexicon provides a good starting point, it lacks hacker asset indicators (e.g., “XSS” (cross-site scripting), “zero-day,” “ransomware,” etc.), which can be useful for hacker asset detection. Hence, we used this lexicon as our initial seed to expand.

**Step 2)** To extend the number of jargon and acronyms related to hacker assets, we expanded this lexicon with three publicly available smaller but more up-to-date lexicons (Arvatz 2017; DarkOwl 2019; Motherboard 2019).

**Step 3)** To add related terms to hacker asset identification in the dark web, two cybersecurity experts scanned 15,000 product descriptions from two large and well-known dark web platforms and added 148 items that did not exist in the previous steps. For example, carding terminology (e.g., “fullz,” “dump,” etc.) was mostly added at this step.

**Step 4)** We added 642 malicious file variant names from New Jersey Cybersecurity and Communications Integration Cell (NJCCIC) (“Cyber Threat Profiles” 2020), including 54 botnets, 94 mobile malware, 234 ransomware, 161 Trojan variants, and 99 other malware types.

**Step 5)** Consulting with a cybersecurity expert, we added hacker asset identifiers that did not exist in the lexicon. As a result, another 55 new frequent hacker asset indicators were added (e.g., “dox,” “Citadel,” “Mirai,” etc.).
**Step 6** To further improve the accuracy of the lexicon, we removed very general terms (e.g., “architecture”) that mainly came from the DHS security lexicon and may appear in contexts other than cybersecurity and indicate non-hacker assets.

**Step 7** We were able to translate the majority of the words into other languages, except for the acronyms such as RAT (Remote Access Trojan). For these acronyms, we used the English acronym since we have observed that English words are sometimes used in foreign languages as well. Also, since English words may be used in foreign languages, we added all English indicators to the Russian, French, and Italian lexicon.

This systematic process yielded a multilingual lexicon in four languages with 1,059 English, 1,625 Russian, 1,512 French, and 1,356 Italian entries). We believe the compiled lexicon could be a helpful resource for cybersecurity firms who want to deploy non-ML-based dark web monitoring, before implementing a fully-fledged ML approach for hacker asset detection. Thus, we published this lexicon as part of our implementation code with the hope that the cybersecurity analytics research community can expand and utilize it in their future work.
