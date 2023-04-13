# English-Sarcasm-Detection
Data Preprocessing
In this section, we load and process data using the SarcasmDataset class. To create a SarcasmDataset, input the path of the data csv file and the tokenizer. Later use pytorch to crate a dataloader for the dataset (in the main script).
Model
In this section, we define the model. We will use the ensemble method, which would use multiple models and combine their outputs to get the final prediction. The models we will use are: 1. Custom LSTM model 2. RoBERTa model 3. BERT model (the 2nd and 3rd model are pretained, so can be change in the future to other pretrained models if needed).

The custom LSTM model is a simple LSTM model is defined in CustomLSTM class. The pretrained models are defined in PretrainedModelPlus class, which can take in any pretrained model and add a hidden layer and output layer on top of it.

Architecture
Custom LSTM
Embedding layer: 50d GloVe embedding
LSTM layer: bidirectional LSTM
Hidden layer: two linear layers with non-linear activation function
Output layer: linear with output size 1
RoBERTa
RoBERTa model: pretrained RoBERTa model, freezed the weights
Hidden layer: two linear layers with non-linear activation function
Output layer: linear with output size 1
BERT
BERT model: pretrained BERT model, freezed the weights
Hidden layer: two linear layers with non-linear activation function
Output layer: linear with output size 1
Ensembling
The models are trained separately and the outputs are combined using combined probability. This is implemented in the predict function.
Evaluation Matrics
We use the f1 score as the evaluation matrics.
Main Script
Instructions for running the main script:

Download the data from here.

Create the dataset and dataloader for each of the models.

Create and Train model

Predict and evaluate f1 score on test set
