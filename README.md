# Topic-Modelling GCN + LDA
This repository explores Latent Dirichlet Allocation methods for text based classification employing various Graph Convolutional Networks.

# Dataset 
Download from:
https://drive.google.com/file/d/10kx3z3bjYFoeRjjg1_DZOAP39Jln0BCh/view?usp=sharing
and keep under TestSGC/ after extracting.

# Step:: 1 Pre-Processing
For pre-processing and arranging the dataset into DataFrame except for 20ng and ohsumed (which are done as given in the code `step_1_data_to_pandas_normal.py`) remaining datasets have their iPyNB in their respective dataset directories.

# Step:: 2 LDA Feature Vector
`step_2_topic_modelling.py`

# Step:: 3 Gathering LDA Feature Vector into a Composite Feature Matrix
For matching the feature matrix of GCN in "Graph Convolutional Networks for Text Classification's" implementation, we have used their file used for indicating document names, training/test split, document labels. Each line is for a document. 
These files are stored under `document_information`.



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

