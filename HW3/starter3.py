import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch.optim as optim


class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(3, 8)
    self.act = nn.ReLU()
    self.output = nn.Linear(8, 3)
# activation function that converts linear results to probabilities
  def softmax(self, outputLayer):
     probabilities = []
     
     for x in outputLayer:
        probs = []
        baseSum = 0
        for feature in x:
            baseSum += math.exp(feature)
        for features in x:
            probs.append(math.exp(features) / baseSum)
        probabilities.append(probs)
     return probabilities

  def forward(self, x):
    x = self.act(self.hidden(x))
    x = self.output(x)
    #x = torch.tensor(x, dtype=torch.float32)
    return x
    

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
                    
            count = count + 1
    return(data)
               
def classify_insurability():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    X = [x[1] for x in train]
    X_test = [x[1] for x in test]
    y = [x[0][0] for x in train]
    Y_test = [x[0][0] for  x in test]
    
    
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_test = sc.fit_transform(X_test)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)

    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    y = ohe.transform(y)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    
    #print(scaled_train)
    model = FeedForward()
    model.train()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    n_epochs = 100
    batch_size = 5
    for epoch in range(n_epochs):
        for i in range(0,len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            #y_pred = torch.tensor(softmax(y_pred))
            ybatch = y[i:i+batch_size]
            l = loss(y_pred, ybatch)
            #print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #print(list(model.parameters()))
        
        print("finished epoch: ", epoch, "loss value: ", l)

    n = 200
    correct = 0
    for i in range(n):
        y_pred = model(X_test[i])
        y_pred = softmax(y_pred)
        y_val = y_pred.index(max(y_pred))
        y_act = Y_test[i]
        if y_val == y_act:
            correct += 1
    proportion = correct/n
    print("% correct: ", proportion)

        
        
    #print(y_pred)
    
    #print(l)
    #print(y_pred)
    
    #print(ff.forward(scaled_train))
    


    # insert code to train simple FFNN and produce evaluation metrics
    
def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv','pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    
def classify_mnist_reg():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    
def classify_insurability_manual():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN

def softmax(outputLayer):
     probabilities = []
     baseSum = 0
     for feature in outputLayer:
        baseSum += math.exp(feature)
     for features in outputLayer:
        probabilities.append(math.exp(features) / baseSum)
     return probabilities
    
def main():
    classify_insurability()
    #classify_mnist()
    #classify_mnist_reg()
    #classify_insurability_manual()
    
if __name__ == "__main__":
    main()


