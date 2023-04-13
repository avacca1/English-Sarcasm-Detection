# English-Sarcasm-Detection
<h2>Data Preprocessing</h2>
We load and process data using the SarcasmDataset class. To create a SarcasmDataset, input the path of the data csv file and the tokenizer. Later use pytorch to crate a dataloader for the dataset (in the main script).

<h2>Model</h2>

We define the model. We will use the ensemble method, which would use multiple models and combine their outputs to get the final prediction. The models we will use are: 
1. Custom LSTM model 
2. RoBERTa model 
3. BERT model (the 2nd and 3rd model are pretained, so can be change in the future to other pretrained models if needed).

The custom LSTM model is a simple LSTM model is defined in CustomLSTM class. The pretrained models are defined in PretrainedModelPlus class, which can take in any pretrained model and add a hidden layer and output layer on top of it.

<h2>Architecture</h2>
<h3>Custom LSTM</h3>

Embedding layer: 50d GloVe embedding

LSTM layer: bidirectional LSTM

Hidden layer: two linear layers with non-linear activation function

Output layer: linear with output size 1

<h3>RoBERTa</h3>

RoBERTa model: pretrained RoBERTa model, freezed the weights

Hidden layer: two linear layers with non-linear activation function

Output layer: linear with output size 1

<h3>BERT</h3>

BERT model: pretrained BERT model, freezed the weights

Hidden layer: two linear layers with non-linear activation function

Output layer: linear with output size 1

<h3>Ensembling</h3>
The models are trained separately and the outputs are combined using combined probability. This is implemented in the predict function.

<h2>Evaluation Matrics</h2>
We use the f1 score as the evaluation matrics.

<h2>Main Script</h2>
Instructions for running the main script:

1. Download the data from here: https://github.com/iabufarha/iSarcasmEval

2. Create the dataset and dataloader for each of the models.

3. Create and Train model

4. Predict and evaluate f1 score on test set
