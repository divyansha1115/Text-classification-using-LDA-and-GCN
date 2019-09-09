# Topic-Modelling + LDA
This repository explores Latent Dirichlet Allocation methods for text based classification employing various Graph Convolutional Networks.

# Dataset 
Ohsumed (http://davis.wpi.edu/xmdv/datasets/ohsumed.html)
A curated text collection from MEDLINE, an online medical information database.

# Features
Our approach is an ensemble work of features from LDA and Word2Vec based approaches.

# Code Description

* LDA features obtained from `topic_modelling.py`
* The data is then converted to LDA probability matrix using `data_to_pandas.py`
* Word2Vec document vectors is extracted using `document_vector.py`
* `SGC/downstream/TextSGC/` contains the model files for training the different network architectures.

The features obtained from LDA together with document vectors are merged in a systematic fashion to obtain a
feature rich map which is then fed into a custom-build GCN model. Experiments were carried on to obtain optimal results.

