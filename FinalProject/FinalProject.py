# final project code goes here

from GenerateJSON import *
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import random
import matplotlib.pyplot as plt
from sklearn import metrics

def assign_rating(y):
    if y > 7:
        return [2]
    if y < 4:
        return [0]
    return [1]


        

class FeedForward_Letterboxd(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(4, 4)
    self.act = nn.ReLU()
    self.output = nn.Linear(4, 3)
    self.softmax = nn.Softmax()
# activation function that converts linear results to probabilities
  

  def forward(self, x):
    x = self.act(self.hidden(x))
    x = self.output(x)
    x = self.softmax(x)
    #x = torch.tensor(x, dtype=torch.float32)
    return x
# ['localized title', 'cast', 'genres', 'runtimes', 'countries', 'country codes', 'language codes', 'color info', 'aspect ratio', 'sound mix', 'certificates', 'original air date', 'rating', 'votes', 'cover url', 'imdbID', 'videos', 'plot outline', 'languages', 'title', 'year', 'kind', 'original title', 'director', 'writer', 'producer', 'composer', 'cinematographer', 'editor', 'casting director', 'production design', 'costume designer', 'make up', 'assistant director', 'art department', 'sound crew', 'special effects', 'visual effects', 'stunt performer', 'camera and electrical department', 'casting department', 'costume department', 'location management', 'transportation department', 'miscellaneous crew', 'akas', 'production companies', 'distributors', 'special effects companies', 'other companies', 'plot', 'synopsis']

def read_user(accountName):
    # reads through the data for each movie reviewed by the specified user in their JSON file and formats it for use in a NN. If a JSON file does not exist for the user, one is created for them
    # 
    # output: a list of examples of the form [[user rating],[features]]

    path = "users/" + accountName + ".json" # user's json files are stored in the users directory

    if not os.path.isfile(path):
        createUserJSON(accountName) # creates a json file for the user if it doesn't exist

    with open(path, 'r') as file:
        data = []
        movies = json.load(file) # a list of all movies reviewed by user
        movie_list = list(movies)
        for movie_name in movie_list:
            movie = movies[movie_name]
            userRating = [movie["userLetterboxdReview"]]
            features = [
                movie["localizedTitle"],
                movie["runtimes"][0],
                movie["genres"][0],
                movie["colorInfo"][0],
                movie["rating"],
                movie["year"],
                movie.get("director", None),
                movie.get("writer", None),
                movie.get("producer", None),
                movie.get("composer", None)
            ]

            for a in movie["topThreeActors"]:
                features.append(a)
            

            data.append([userRating,features])
        
    return data

def read_user_numeric(accountName):
    # reads through the data for each movie reviewed by the specified user in their JSON file and formats it for use in a NN. If a JSON file does not exist for the user, one is created for them
    # 
    # output: a list of examples of the form [[user rating],[features]]

    path = "users/" + accountName + ".json" # user's json files are stored in the users directory

    if not os.path.isfile(path):
        createUserJSON(accountName) # creates a json file for the user if it doesn't exist

    with open(path, 'r') as file:
        data = []
        movies = json.load(file) # a list of all movies reviewed by user
        movie_list = list(movies)
        genreList = []
        for movie_name in movie_list:
            movie = movies[movie_name]
            userRating = [movie["userLetterboxdReview"]]
            features = [
                movie["runtimes"][0],
                movie["rating"],
                movie["year"],
            ]
            genreList.append(movie["genres"][0])

            data.append([userRating,features])
        
    return data, genreList

def run_nn():
    data, genreList = read_user_numeric("schaffrillas")
    random.shuffle(data)
    genreList = [[x] for x in genreList]
    le = OrdinalEncoder()

    genreList = le.fit_transform(genreList)

    print(genreList)

    for i in range(len(data)):
        data[i][1].append(genreList[i][0])
    print(data)

    X = [x[1] for x in data]
    Y = [assign_rating(x[0][0]) for x in data]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(Y)
    Y = ohe.transform(Y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    #np.concatenate(X,genreList)
    print(X)
    print(Y)
    x_train = X[:500]
    x_test = X[500:]
    y_train = Y[:500]
    y_test = Y[500:]

    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)

    model = FeedForward_Letterboxd()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loss, train_accuracy = [],[]

    n_epochs = 15
        
    for epoch in range(n_epochs):
        # set model to training mode
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        # Train network one observation at a time
        for i in range(0,len(x_train)):
            Xbatch = x_train[i]
            y_pred = model(Xbatch)
            y_real = y_train[i]
            l = loss(y_pred, y_real)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            # determine training loss
            total_loss += l.item()
            # determine training accuracy
            total_correct += (torch.argmax(y_pred) == torch.argmax(y_real)).item()
            total_samples += 1

        
        # populate training data arrays for learning functions
        train_loss.append(total_loss / len(X))
        train_accuracy.append(total_correct / total_samples)
        print("finished epoch: ", epoch, "loss value: ", total_loss / len(X), " Percentage Correct: ", total_correct / total_samples)
    n = len(x_test)
    correct = 0
    predicted = [0] * n # 1D array with the prediction for each example
    actual = [0] * n
    for i in range(n):
        y_pred = model(x_test[i])
        predicted[i] = torch.argmax(y_pred).item()
        actual[i] = torch.argmax(y_test[i]).item()
        if (torch.argmax(y_pred) == torch.argmax(y_test[i])).item():
            correct += 1
    proportion = correct/n
    print("% correct: ", proportion, n)  

    f1 = metrics.f1_score(actual,predicted, average = "weighted")
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2])
    cm_display.plot()
    title = "% Correct: " + str(proportion) + " | F1 Score: " + str(f1)
    plt.title(title)
    plt.show()




def format_for_collaborative_filtering(accountName):
    path = "users/"
    
    data = []
    
    movie_axis = []

    for filename in os.listdir(path): 
        
        user_path = path + filename
        
        with open(user_path, 'r') as file:


            movies = json.load(file) # a list of all movies reviewed by user
            movie_list = list(movies)
            
            if len(movie_axis) == 0:
                movie_axis.append(movie_list)

            ratings = []

            for movie_name in movie_list:
                movie = movies[movie_name]
                userRating = [movie["userLetterboxdReview"]]
                ratings.append(userRating)


            data.append(ratings)


                
            
    
    return data

#format_for_collaborative_filtering(accountName = "nmcassa")

run_nn()