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

def run_nn(username):
    data, genreList = read_user_numeric(username)
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

# returns Cosine Similarity between vectors a dn b
def cosim(a,b, knn = True):
    dist = 0
    if knn:
        dist = sum(int(x)*int(y) for x, y in zip(a, b)) /(vecSumSqrt(a) * vecSumSqrt(b))
    else:
        dist = sum(int(x)*int(y) for x, y in zip(a, b)) /(1+(vecSumSqrt(a) * vecSumSqrt(b)))

    return(dist)

# gets the square root of the sum of the vector
def vecSumSqrt(vec):
    dist = 0
    for i in range(len(vec)):
        dist += int(vec[i]) ** 2
    return dist ** 0.5

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (float(a[i]) - float(b[i])) ** 2
    dist = dist ** 0.5
    return(dist)

# use collaborative filtering to suggest movies to the user
def get_knn_collaborative_filtering(accountName, k, metric):
    path = "users/"
    accountPath = path + accountName + ".json"
    userSimilarity = []
    print("[FinalProject] Determining similarity for user " + accountName)
    # creates a json file for the user if it doesn't exist
    if not os.path.isfile(accountPath):
        createUserJSON(accountName) 

    # populate user movie list
    with open(accountPath, 'r') as file:
        userMovies = json.load(file) # a list of all movies reviewed by main user

    # a list of movies that have not been seen by the main user but have been seen by other users
    unseen_movies = []

    # find similarity between users [userName, manhattan distance, euclidean distance, cosine similarity, number of reviews in common]
    # The similarity is found by taking the difference in rating between each movie reviews in common, and then summing all the differences and dividing by the total number of reviews in common
    for filename in os.listdir(path): 
        user_path = path + filename
        # load user if it is not current account
        if user_path != accountPath:
            with open(user_path, 'r') as file:
                manhattanDistance = 0
                userArray = []
                otherUserArray = []
                numberInCommon = 0
                currMovies = json.load(file) # a list of all movies reviewed by other user
                # determine manhattan distance and prep array for cosine similarity
                for movie in currMovies:
                    if movie in userMovies:
                        numberInCommon += 1
                        manhattanDistance += abs(userMovies[movie]["userLetterboxdReview"] - currMovies[movie]["userLetterboxdReview"])
                        userArray.append(userMovies[movie]["userLetterboxdReview"])
                        otherUserArray.append(currMovies[movie]["userLetterboxdReview"])
                    else:
                        unseen_movies.append(movie)


                # decided to loop thru movies other people have seen so that we can create a list of movies the main user hasn't seen in order to pull from a list of possible recommendations

                # for movie in userMovies:
                #     if movie in currMovies:
                #         numberInCommon += 1
                #         manhattanDistance += abs(userMovies[movie]["userLetterboxdReview"] - currMovies[movie]["userLetterboxdReview"])
                #         userArray.append(userMovies[movie]["userLetterboxdReview"])
                #         otherUserArray.append(currMovies[movie]["userLetterboxdReview"])
                
                
                # normalize arrays for distance metrics
                # userArray = np.array(userArray)
                # otherUserArray = np.array(otherUserArray)

                # userArray = userArray - userArray.mean()
                # otherUserArray = otherUserArray - otherUserArray.mean()

                # maxAbsUser = np.max(np.abs(userArray))
                # maxAbsOther = np.max(np.abs(otherUserArray))

                # userArray = userArray / maxAbsUser
                # otherUserArray = otherUserArray / maxAbsOther
                # calculate cosine similarity
                cosineSimilarity = cosim(userArray, otherUserArray)
                euclideanDistance = euclidean(userArray, otherUserArray)
                # add similarity to list
                userSimilarity.append([user_path, manhattanDistance / numberInCommon, euclideanDistance / numberInCommon, cosineSimilarity, numberInCommon])
                print("[FinalProject] Similarity determined for user " + filename)

    # sort in ascending order
    if metric == 1:
        userSimilarity = sorted(userSimilarity, key=lambda x: x[1])
    elif metric == 2:
        userSimilarity = sorted(userSimilarity, key=lambda x: x[2])
    else:
        userSimilarity = sorted(userSimilarity, key=lambda x: x[3], reverse=True)

    return userSimilarity[:k], unseen_movies


def recommend_N_movies(N, knn, unseen_movies, metric):
    #(Metric: 1=manhattan, 2=euclidean, 3=cosine)

    # preload dictionaries with neccesary info from json to optimize runtime
    neighborsMovies = dict()
    neighborsMetrics = dict()
    for n in knn:
        with open(n[0], 'r') as file:
            neighborsMovies[n[0]] = json.load(file)
            neighborsMetrics[n[0]] = n[metric] 

    i = 1

    recommendations = {}

    for movie in unseen_movies:
        #print(str(round(i/len(unseen_movies)*100, 2)) + "% Completed")
        
        recommendation_score = 0

        num_scores = 0

        for neighbor in neighborsMetrics: # loop through the K nearest neighbors
            similarity_score = neighborsMetrics[neighbor]    
            curr_user_movies = neighborsMovies[neighbor]      

            if movie in curr_user_movies:
                curr_user_rating = curr_user_movies[movie]["userLetterboxdReview"]
                
                if metric == 3:
                    curr_score = similarity_score * curr_user_rating
                else:
                    curr_score = similarity_score * (11 - curr_user_rating)

                recommendation_score += curr_score

                num_scores += 1
        
        if num_scores > 0:
            recommendations[movie] = recommendation_score / num_scores
    
        i += 1


    if metric == 3:
        recommendations = sorted(recommendations.items(), key=lambda x:x[1], reverse=True)
    else:
        recommendations = sorted(recommendations.items(), key=lambda x:x[1])

    return recommendations[:N]

# Conduct collaborative filtering algorithm to recommend movies
def collaborative_filtering(accountName):
    # get nearest neighbors (Metric: 1=manhattan, 2=euclidean, 3=cosine)

    metric = 1

    nearestNeighbors,unseen_movies = get_knn_collaborative_filtering(accountName, k=5, metric=metric)
    print("[FinalProject] Nearest Neighbors:", nearestNeighbors)

    N = 5

    recommendations = recommend_N_movies(N, nearestNeighbors, unseen_movies, metric = metric)
    print("[FinalProject] Top " + str(N) + " recommended movies are: ")
    print(recommendations)

# runner
username = input("[FinalProject] Please enter your Letterboxd username: ")
selection = input("[FinalProject] Would you like to (1) get movie reccomendations through collaborative filtering or (2) train a neural network on your watched movies? ")
if (selection == "1"):
    collaborative_filtering(username)
elif (selection == "2"):
    run_nn(username)
else:
    print("[FinalProject] You did not enter (1) or (2)")
