# English-Sarcasm-Detection

https://github.com/avacca1/English-Sarcasm-Detection


<h2>Data Preprocessing</h2>
We define the SarcasmDataset class, which we use to load and pre-process the sarcasm data. This includes encoding emojis as text tokens instead of their unicode representations, and convert usernames and urls to single token. To create a SarcasmDataset, input the path of the data csv file and the tokenizer. We then use pytorch to crate a dataloader for the dataset (can be found in the main script).

<h2>Models</h2>

To complete the task we use the ensemble method, which trains multiple models and combines their outputs to get a final prediction. The models we will use are:
1. finiteautomata/bertweet-base-sentiment-analysis
2. pysentimiento/bertweet-irony
3. cardiffnlp/bertweet-base-irony

Each model is based upon BERTweet, a RoBERTa model trained on English Tweets. [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) is an iteration of BERT that modifies some key hyperparameters, removes the next sentence pretraining objective, and is trained with larger mini-batches and higher learning rates.

The pretrained models are defined in PretrainedModelPlus class, which can take in any pretrained model and add a hidden layer and output layer on top of it in order to classify it's output.

<h2>Model Descriptions</h2>

### [finiteautomata/bertweet-base-sentiment-analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)


#### Paper: 
Title: pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks

author: Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque

year: 2021

eprint: 2106.09462


RoBERTa model trained on English tweets for the purpose of sentiment analysis. Trained on SemEval 2017 corpus. Uses Pos, Neg, Neu labels. 

### [pysentimiento/bertweet-irony](https://huggingface.co/pysentimiento/bertweet-irony)

RoBERTa model trained with SemEval 2018 dataset Task 3. Based on BERTweet. 

### [cardiffnlp/bertweet-base-irony](https://huggingface.co/cardiffnlp/bertweet-base-irony)



RoBERTa model similar to bertweet irony, but with fewer encoder layers. 

<h3>Linear Layer</h3>

Single linear layer to fine tune output of model. Input dimension is the size of the hidden state of the pretrained model.This is followed by a linear layer of size 64. The output dimension is of size 1 for classification.

<h3>Ensembling</h3>
The models are trained separately, and the outputs are combined by averaging their probability. This average is then compared to the threshold for classificaiton. This is all implemented in the predict function.

## Evaluation Metrics
We use the f1 score as the evaluation matrics. F1 score is computed for each individual model as well as the final result.

## Result/Output

The output of each model (model 1, model 2, model 3 / bertweet-base-sentiment-analysis, bertweet-irony, bertweet-base-irony), are labeled output-1.csv, output-2.csv, output-3.csv respectively. The ensemble output file is output-123.csv.

**Final Result**: F1=50.3 (ensemble of 3 models, output-123.csv)

## Main Script Instructions (More details in notebook)

### 1. Download the data from [here](https://github.com/iabufarha/iSarcasmEval).

### 2. Run all cells below Main Script Header. Script does the following:

  Creates the dataset and dataloader for each of the models.
  
  Initializes the three models.
  
  a. Trains the three models OR b. Loads the three previously trained and continue to fine tune models (depends on presence of files model1.pt, model2.pt, model3.pt).

  Predicts and evaluate f1 score on test set for each individual model (just for comparison).

  Predicts and evaluate f1 score on test set for ensemble of models.
