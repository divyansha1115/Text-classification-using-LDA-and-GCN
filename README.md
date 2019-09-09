# Topic-Modelling GCN + LDA
This repository explores Latent Dirichlet Allocation methods for text based classification employing various Graph Convolutional Networks.

# Dataset 
Ohsumed (http://davis.wpi.edu/xmdv/datasets/ohsumed.html)
A curated text collection from MEDLINE, an online medical information database.

# Features
Our approach is an ensemble work of features from LDA and Graph Convolution.

# Code Description

* LDA features obtained from `topic_modelling.py`
* The data is then converted to LDA probability matrix using `data_to_pandas.py`
* The GCN features are calculated considering various parameters and stored for training.
* `SGC/downstream/TextSGC/` contains the model files for training the different network architectures.
* The model also involves the use of skip-architecture which improves model performance.

The features obtained from LDA together with the GCN features are merged in a systematic fashion to obtain a
feature rich map which is then fed into a custom-build model. Experiments were carried on to obtain optimal results.

