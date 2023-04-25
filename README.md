# English-Sarcasm-Detection
<h2>Data Preprocessing</h2>
We define the SarcasmDataset class, which we use to load and pre-process the sarcasm data. This includes encoding emojis as text tokens instead of their unicode representations. To create a SarcasmDataset, input the path of the data csv file and the tokenizer. We then use pytorch to crate a dataloader for the dataset (can be found in the main script).

<h2>Models</h2>

To complete the task we use the ensemble method, which trains multiple models and combines their outputs to get a final prediction. The models we will use are:
1. finiteautomata/bertweet-base-sentiment-analysis
2. pysentimiento/bertweet-irony
3. cardiffnlp/bertweet-base-irony

Each model is based upon BERTweet, a RoBERTa model trained on English Tweets. [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) is an iteration of BERT that modifies some key hyperparameters, removes the next sentence pretraining objective, and is trained with larger mini-batches and higher learning rates.

The pretrained models are defined in PretrainedModelPlus class, which can take in any pretrained model and add a hidden layer and output layer on top of it in order to classify it's output.

<h2>Model Descriptions</h2>

### [bertweet-base-sentiment-analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)


Paper: 
Title: pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks
author: Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
year={2021},
eprint={2106.09462},
archivePrefix={arXiv},
primaryClass={cs.CL}




RoBERTa model trained on English tweets for the purpose of sentiment analysis. Trained on SemEval 2017 corpus. Uses Pos, Neg, Neu labels. 

### [pysentimiento/bertweet-irony](https://huggingface.co/pysentimiento/bertweet-irony)



RoBERTa model trained with SemEval 2018 dataset Task 3. Based on BERTweet. Positive marks irony, negative marks non ironic.

### [cardiffnlp/bertweet-base-irony](https://huggingface.co/cardiffnlp/bertweet-base-irony)



RoBERTa model similar to bertweet irony, but with fewer encoder layers. 

<h3>Linear Layer</h3>

Single linear layer to fine tune output of model. Input dimension is the size of the hidden state of the pretrained model.This is followed by a linear layer of size 64. The output dimension is of size 1 for classification.

<h3>Ensembling</h3>
The models are trained separately, and the outputs are combined by averaging their probability. This average is then compared to the threshold for classificaiton. This is all implemented in the predict function.

<h2>Evaluation Metrics</h2>
We use the f1 score as the evaluation matrics. F1 score is computed for each individual model as well as the final result.

<h2>Main Script</h2>
Instructions for running the main script:

1. Download the data from here: https://github.com/iabufarha/iSarcasmEval

2. Create the dataset and dataloader for each of the models.

3. Create and Train model

4. Predict and evaluate f1 score on test set
