# NLP-A-3
## Overview
### This project is a demonstration of how to perform sentiment analysis using Recurrent Neural Networks (RNNs) by fusing word embeddings from Word2Vec and FastText models.

## Requirements
### Python 3.x
### PyTorch
### Gensim
### NumPy
### scikit-learn

## Installation
### You can install the required Python libraries using pip:

## RNN
### Download the pre-trained Word2Vec and FastText models and place them in the project directory with the names w2v_model.bin and ft_model.bin respectively.

## Implementation Details
### Data Preparation
#### CustomDataset: This class prepares the dataset for training the RNN model. It takes texts, labels, word2idx, word_embeddings_1, word_embeddings_2, and max_length as input parameters.
#### train_rnn_model: This function is responsible for training the RNN model. It initializes the model, optimizer, and loss function. It also prepares word to index mapping, creates datasets, and dataloaders. The model is trained using a training loop and then evaluated for both Word2Vec and FastText embeddings.

## RNN Model
### The RNN model used here is a simple classifier with the following architecture:
#### Input size: Sum of the dimensions of Word2Vec and FastText embeddings.
#### Hidden size: 128
#### Output size: 2 (Positive or Negative sentiment)

## Evaluation Metrics
### The following evaluation metrics are used to evaluate the RNN model:

#### Accuracy
#### Precision
#### Recall
#### F1 Score


