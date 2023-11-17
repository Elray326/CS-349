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
import matplotlib.pyplot as plt
from sklearn import metrics


class FeedForward_Insurability(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(3, 8)
    self.act = nn.ReLU()
    self.output = nn.Linear(8, 3)
# activation function that converts linear results to probabilities
  

  def forward(self, x):
    x = self.act(self.hidden(x))
    x = self.output(x)
    #x = torch.tensor(x, dtype=torch.float32)
    return x
class FeedForward_Mnist(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(784, 24)
    self.act = nn.ReLU()
    self.output = nn.Linear(24, 10)  

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
        # apply binary transformation
        data_set = binaryConverter(data_set)
    return(data_set)

# converts the data set to be a hard coded 1 or 0 depending on threshold
def binaryConverter(data):
    # set border value
    borderValue = 120
    # iterate through data and convert to 1 or 0
    for i in range(len(data)):
        for j in range(len(data[0][1])):
            if int(data[i][1][j]) < borderValue:
                data[i][1][j] = 0
            else:
                data[i][1][j] = 1
    return data
        
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
    X_valid = [x[1] for x in valid]
    y = [x[0][0] for x in train]
    Y_test = [x[0][0] for  x in test]
    Y_valid = [x[0][0] for x in valid]
    
    
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_test = sc.fit_transform(X_test)
    X_valid = sc.fit_transform(X_valid)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
    X_valid = torch.tensor(X_valid, dtype=torch.float32, requires_grad=True)

    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32).reshape(-1, 1)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    y = ohe.transform(y)
    Y_valid = ohe.transform(Y_valid)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32, requires_grad=True)
    
    #print(scaled_train)
    model = FeedForward_Insurability()
    #model.train()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # keep track of learning curves for training and validation data sets
    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
    n_epochs = 20
    
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        # Train network one observation at a time
        for i in range(0,len(X)):
            Xbatch = X[i]
            y_pred = model(Xbatch)
            #y_pred = torch.tensor(softmax(y_pred))
            ybatch = y[i]
            l = loss(y_pred, ybatch)
            #print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #print(list(model.parameters()))
            
            # determine training loss
            total_loss += l.item()
            # determine training accuracy
            total_correct += (torch.argmax(y_pred) == torch.argmax(ybatch)).item()
            total_samples += 1

        # Run with validation data (TODO: should we softmax)
        model.eval()
        with torch.no_grad():
            pred = model(X_valid)
            val_loss.append(loss(pred, Y_valid).item())
            val_accuracy.append((torch.argmax(Y_valid, dim=1) == torch.argmax(pred, dim=1)).sum().item() / pred.size(0))
        
        # populate training data arrays for learning functions
        train_loss.append(total_loss / len(X))
        train_accuracy.append(total_correct / total_samples)
        print("finished epoch: ", epoch, "loss value: ", total_loss / len(X), " Percentage Correct: ", total_correct / total_samples)  

    # determine accuracy
    n = len(X_test)
    correct = 0
    predicted = [0] * n # 1D array with the prediction for each example
    for i in range(n):
        y_pred = model(X_test[i])
        y_pred = softmax(y_pred)
        y_val_predicted = y_pred.index(max(y_pred))
        y_act = Y_test[i]
        predicted[i] = y_val_predicted
        if y_val_predicted == y_act:
            correct += 1
    proportion = correct/n
    print("% correct: ", proportion)

    actual = Y_test

    # generate and display confusion matrix
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Bad", "Neutral", "Good"])
    cm_display.plot()
    title = "% Correct: " + str(proportion)
    plt.title(title)
    plt.show()

    # generate learning curves
    plt.plot(train_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss VS. Epochs Training")
    plt.show()
    plt.plot(train_accuracy)
    plt.title("Accuracy VS. Epochs Training")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()    
    plt.plot(val_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss VS. Epochs Validation")
    plt.show()
    plt.plot(val_accuracy)
    plt.title("Accuracy VS. Epochs Validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()    
    #print(y_pred)
    
    #print(l)
    #print(y_pred)
    
    #print(ff.forward(scaled_train))
    


    # insert code to train simple FFNN and produce evaluation metrics
    
def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')


    X = [x[1] for x in train]
    X_test = [x[1] for x in test]
    X_valid = [x[1] for x in valid]
    y = [float(x[0][0]) for x in train]
    
    Y_test = [float(x[0][0]) for  x in test]
    Y_valid = [float(x[0][0]) for x in valid]
    
    
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_test = sc.fit_transform(X_test)
    X_valid = sc.fit_transform(X_valid)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    print("x features: ", len(X[0]))
    X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
    X_valid = torch.tensor(X_valid, dtype=torch.float32, requires_grad=True)

    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32).reshape(-1, 1)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    y = ohe.transform(y)
    print(y)
    Y_valid = ohe.transform(Y_valid)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32, requires_grad=True)
    #show_mnist('mnist_test.csv','pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    model = FeedForward_Mnist()
    #model.train()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # keep track of learning curves for training and validation data sets
    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
    
    n_epochs = 20
    
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        # Train network one observation at a time
        for i in range(0,len(X)):
            Xbatch = X[i]
            y_pred = model(Xbatch)
            #y_pred = torch.tensor(softmax(y_pred))
            ybatch = y[i]
            l = loss(y_pred, ybatch)
            #print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #print(list(model.parameters()))
            
            # determine training loss
            total_loss += l.item()
            # determine training accuracy
            total_correct += (torch.argmax(y_pred) == torch.argmax(ybatch)).item()
            total_samples += 1

         # Run with validation data (TODO: should we softmax)
        model.eval()
        with torch.no_grad():
            pred = model(X_valid)
            val_loss.append(loss(pred, Y_valid).item())
            val_accuracy.append((torch.argmax(Y_valid, dim=1) == torch.argmax(pred, dim=1)).sum().item() / pred.size(0))
        
        # populate training data arrays for learning functions
        train_loss.append(total_loss / len(X))
        train_accuracy.append(total_correct / total_samples)
        print("finished epoch: ", epoch, "loss value: ", total_loss / len(X), " Percentage Correct: ", total_correct / total_samples)  
    
    # generate learning curves
    plt.plot(train_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss VS. Epochs Training")
    plt.show()
    plt.plot(train_accuracy)
    plt.title("Accuracy VS. Epochs Training")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()    
    plt.plot(val_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss VS. Epochs Validation")
    plt.show()
    plt.plot(val_accuracy)
    plt.title("Accuracy VS. Epochs Validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()    

    
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

# activation function that converts linear results to probabilities
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


