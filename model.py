# -*- coding:utf-8 -*-
"""
This py file is a skeleton for the main project components.
Places that you need to add or modify code are marked as TODO
"""

import torch
import torch.nn as nn
import torch.optim
import numpy as np


def word2index(word, vocab):
    """
    Convert a word token to a dictionary index
    """
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert a dictionary index to a word token
    """
    for w, v in vocab.items():
        if v[0] == index:
            return w
    return 0

class Model(object):
    def __init__(self, args, vocab, trainlabels, trainsentences, testlabels, testsentences):
        """ The Text Classification model constructor """
        self.embeddings_dict = {}
        self.datarep = args.datarep
        if self.datarep == "GLOVE":
            print("Now we are using the GloVe embeddings")
            self.load_glove(args.embed_file)
        else:
            print("Now we are using the BOW representation")
        self.vocab = vocab
        self.trainlabels = trainlabels
        self.trainsentences = trainsentences
        self.testlabels = testlabels
        self.testsentences = testsentences
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.traindataset = []
        self.testdataset = []

        """
        TODO
        You should modify the code for the baseline classifiers for self.datarep
        shown below, which is a three layer model with an input layer, a hidden layer,
        and an output layer. You will need at least to define the dimensions for
        the size of the input layer (ISIZE; see where this is passed in by argparse),
        and the hidden layer (e.g., HSIZE).  Do not change the size of the output
        layer, which is currently 2, as this corresponds to the number of sentiment classes.
        You need to choose an activation function. Once you get this working
        by uncommenting these lines, adding the activation function, and replacing
        ISIZE and HSIZE, see if you can achieve the classification accuracy on movies
        of 0.85 for the GLoVE representation, or 0.90 for the BOW representation.  
        You are free to modify the code for self.model, e.g., to add more hidden layers, or 
        to change the input representation created in prepare_datasets(), to raise the accuracy.
        """

        ISIZE_BOW = len(self.vocab)
        HSIZE_BOW = self.hidden_size

        ISIZE_GLOVE = self.embed_size
        HSIZE_GLOVE = 2*self.hidden_size

        if self.datarep == "GLOVE":
            self.model = nn.Sequential(
                nn.Linear(ISIZE_GLOVE, HSIZE_GLOVE),
                nn.Sigmoid(),
                nn.Linear(HSIZE_GLOVE, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_size, 2),
                nn.LogSoftmax(),)
        else:
            self.model = nn.Sequential(
                nn.Linear(ISIZE_BOW, HSIZE_BOW),
                nn.Tanh(),
                nn.Linear(HSIZE_BOW, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, 2),
                nn.LogSoftmax(), )

    def prepare_datasets(self):
        """
        Load both training and test
        Convert the text spans to BOW or GLOVE vectors
        """

        datasetcount = 0

        for setOfsentences in [self.trainsentences, self.testsentences]:

            sentcount = 0
            datasetcount += 1

            for sentence in setOfsentences:
                sentcount += 1
                # vsentence holds lexical (GLOVE) or word index (BOW) input to sentence2vec
                vsentence = []
                for l in sentence:
                    if l in self.vocab:
                        if self.datarep == "GLOVE":
                            vsentence.append(l)
                        else:
                            vsentence.append(word2index(l, self.vocab))
                svector = self.sentence2vec(vsentence, self.vocab)
                if (len(vsentence) > 0) & (datasetcount == 1): # train
                    self.traindataset.append(svector)
                elif (len(vsentence) > 0) & (datasetcount == 2): # test
                    self.testdataset.append(svector)

        print("\nDataset size for train: ",len(self.traindataset)," out of ",len(self.trainsentences))
        print("\nDataset size for test: ",len(self.testdataset)," out of ",len(self.testsentences))
        indices = np.random.permutation(len(self.traindataset))

        self.traindataset = [self.traindataset[i] for i in indices]
        self.trainlabels = [self.trainlabels[i] for i in indices]
        self.trainsentences = [self.trainsentences[i] for i in indices]

    def rightness(self, predictions, labels):
        """
        Error rate
        """
        pred = torch.max(predictions.data, 1)[1]
        rights = pred.eq(labels.data.view_as(pred)).sum()
        return rights, len(labels)

    def sentence2vec(self, sentence, dictionary):
        """
        Convert sentence text or indices to vector representation
        """
        if self.datarep == "GLOVE":
            sentence_vector = [0] * self.embed_size
            countWord = 0

            for word in sentence:
                if word in self.embeddings_dict:
                    sentence_vector += self.embeddings_dict[word]
                    countWord += 1
            sentence_vector = sentence_vector // countWord

            return sentence_vector
        else:
            sentence_vector = [0] *len(dictionary)
            for i in sentence:
                sentence_vector[i] += 1
            return sentence_vector


    def load_glove(self, path):
        """
        Load Glove embeddings dictionary
        """
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        return 0

    def training(self):
        """
            The training and testing process.
        """
        losses = []

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr)

        if self.datarep == "GLOVE":
            tr_epochs = 10
        else:
            tr_epochs = 10

        for epoch in range(tr_epochs):
            print(epoch)
            for i, data in enumerate(zip(self.traindataset, self.trainlabels)):
                x, y = data
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                optimizer.zero_grad()
                # predict
                predict = self.model(x)
                # calculate loss
                loss = loss_function(predict, y)
                losses.append(loss.data.numpy())
                loss.backward()
                optimizer.step()
                # test every 1000 data
                if i % 1000 == 0:
                    test_losses = []
                    rights = []
                    for j, test in enumerate(zip(self.testdataset, self.testlabels)):
                        x, y = test
                        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                        y = torch.tensor(np.array([y]), dtype=torch.long)
                        predict = self.model(x)
                        right = self.rightness(predict, y)
                        rights.append(right)
                        loss = loss_function(predict, y)
                        test_losses.append(loss.data.numpy())

                    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                    print('At epoch {}: Training loss: {:.2f}, Test loss: {:.2f}, Test Acc: {:.2f}'.format(epoch, np.mean(losses),
                                                                                                           np.mean(test_losses), right_ratio))
        print("End Testing/Training")
