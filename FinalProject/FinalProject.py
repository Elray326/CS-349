# final project code goes here

from GenerateJSON import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import Counter


# Neural Network class
class FeedForward_Letterboxd(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.hidden = nn.Linear(input_size, input_size)
    self.act = nn.ReLU()
    self.output = nn.Linear(input_size, 3)
    self.softmax = nn.Softmax()  

  def forward(self, x):
    x = self.act(self.hidden(x))
    x = self.output(x)
    x = self.softmax(x)
    #x = torch.tensor(x, dtype=torch.float32)
    return x
  
def assign_rating(y):
    if y > 7:
        return [2]
    if y < 4:
        return [0]
    return [1]
def mean(arr):
    return sum(arr) / len(arr)

def mode(arr):
    count = Counter(arr)
    max_count = max(count.values())
    mode = [k for k, v in count.items() if v == max_count]
    return mode

def calculate_variance(data):
    mean_value = sum(data) / len(data)
    variance = sum((x - mean_value) ** 2 for x in data) / len(data)
    return variance

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
        print("[FinalProject] " + path + " does not exist. Creating JSON...")
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

    ohe_rand = OneHotEncoder(sparse=False)
    # create one hot encoding for genres
    ohe_genres = OneHotEncoder(sparse=False)
    genreList = ohe_genres.fit_transform([[x] for x in genreList])

    # Prepare X and Y
    X = [x[1] for x in data]
    Y = [assign_rating(x[0][0]) for x in data]

    # One-hot encode Y if it's categorical
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    Y = np.array(Y).reshape(-1, 1)
    Y = ohe.fit_transform(Y)

    # Scale X
    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    # Concatenate genre encoding with other features
    X = np.array([np.concatenate((x, genreList[i])) for i, x in enumerate(X)])
    Y = np.array(Y)
    input_size = X.shape[1]

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 


    # PyTorch Data Loaders
    batch_size = 2
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model, Loss, and Optimizer
    model = FeedForward_Letterboxd(input_size)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)

    # Training
    n_epochs = 15
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for Xbatch, y_real in train_loader:
            y_pred = model(Xbatch)
            l = loss(y_pred, y_real)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l.item()
            total_correct += (torch.argmax(y_pred, dim=1) == torch.argmax(y_real, dim=1)).sum().item()
            total_samples += y_real.size(0)

        print(f"Epoch: {epoch}, Loss: {total_loss / total_samples}, Accuracy: {total_correct / total_samples}")

    # Testing
    model.eval()
    correct, total, rcorrect = 0, 0, 0
    predicted, actual, r_arr = [], [],[]
    
    with torch.no_grad():
        for Xbatch, y_real in test_loader:
            print(y_real)
            y_pred = model(Xbatch)
            predicted.extend(torch.argmax(y_pred, dim=1).tolist())
            actual.extend(torch.argmax(y_real, dim=1).tolist())
            correct += (torch.argmax(y_pred, dim=1) == torch.argmax(y_real, dim=1)).sum().item()
            total += y_real.size(0)
    r_arr = [random.randint(0,2) for x in range(len(predicted))]
    for i in range(len(r_arr)):
        rcorrect += r_arr[i] == actual[i]

    proportion = correct / total
    r_proportion = rcorrect / total
    print(f"% correct: {proportion}")

    # Metrics
    f1 = metrics.f1_score(actual, predicted, average="weighted")
    rf1 = metrics.f1_score(actual, r_arr, average="weighted")
    #rf1 = metrics.f1_score(actual, r_arr,average="weighted")
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["negative", "neutral", "positive"])
    cm_display.plot()
    plt.title(f"% Correct: {proportion} | F1 Score: {f1}")
    plt.show()

    confusion_matrix2 = metrics.confusion_matrix(actual, r_arr)
    cm_display2 = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix2, display_labels=["negative", "neutral", "positive"])
    cm_display2.plot()
    plt.title(f"% Correct: {r_proportion} | F1 Score: {rf1}")
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
    # in case the vector randomly adds up to 0
    if dist == 0:
        dist = 1
    return dist ** 0.5

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (float(a[i]) - float(b[i])) ** 2
    dist = dist ** 0.5
    return(dist)

# use collaborative filtering to suggest movies to the user
def get_knn_collaborative_filtering(accountName, k, metric, accuracyTestMode):
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
    removed_movies = dict()

    # to check accuracy remove 20% of movies
    if accuracyTestMode == True:
        numberToRemove = int(0.2 * len(userMovies))
        moviesToRemove = random.sample(list(userMovies.keys()), numberToRemove)
        removed_movies = {key: userMovies[key]["userLetterboxdReview"] for key in moviesToRemove}
        for key in moviesToRemove:
            del userMovies[key]

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

                # Ensure that users have seen at least 10 movies in common
                if numberInCommon < 10: 
                    numberInCommon = 1
                    manhattanDistance = 100000000
                    euclideanDistance = 100000000
                    cosineSimilarity = -1
                else:
                    # calculate cosine similarity and euclidean distance
                    cosineSimilarity = cosim(userArray, otherUserArray)
                    euclideanDistance = euclidean(userArray, otherUserArray)

                # add similarity to list
                userSimilarity.append([user_path, manhattanDistance / numberInCommon, euclideanDistance / numberInCommon, cosineSimilarity, numberInCommon])
                print("[FinalProject] Similarity determined for user " + filename)
    
    # remove duplicates from unseen movies array 
    unseen_movies = set(unseen_movies)

    # sort in ascending order
    if metric == 1:
        userSimilarity = sorted(userSimilarity, key=lambda x: x[1])
    elif metric == 2:
        userSimilarity = sorted(userSimilarity, key=lambda x: x[2])
    else:
        userSimilarity = sorted(userSimilarity, key=lambda x: x[3], reverse=True)

    return userSimilarity[:k], unseen_movies, removed_movies

def recommend_N_movies(N, knn, unseen_movies, metric, accuracyTestMode):
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

                # to test average score of all ratings without them being skewed for accuracy
                if accuracyTestMode == True: 
                    curr_score = curr_user_rating
                elif metric == 3: #cosim
                    curr_score = similarity_score * curr_user_rating
                else: #manhattan or euclidean
                    weight = 1 / (similarity_score + 1)
                    curr_score = weight * curr_user_rating

                recommendation_score += curr_score
                num_scores += 1

                # add weight to number of reviews if not in accuracy mode
                if accuracyTestMode == True or num_scores == 1:
                    divideByConstant = num_scores
                else:
                    divideByConstant = num_scores - ((num_scores * 3) / 100)
        
        if num_scores > 0:
            recommendations[movie] = recommendation_score / divideByConstant
    
        i += 1


    if metric == 3:
        recommendations = sorted(recommendations.items(), key=lambda x:x[1], reverse=True)
    else:
        recommendations = sorted(recommendations.items(), key=lambda x:x[1])

    if accuracyTestMode:
        return recommendations
    return recommendations[:N]

# deduce accuracy of collaborative filtering
def calculateAccuracyTestResults(removed_movies, recommendations):
    recDict = dict(recommendations)
    combined_ratings = {}
    correct = 0
    for movie, userRating in removed_movies.items():
        if movie in recDict:
            recRating = recDict[movie]
            combined_ratings[movie] = (userRating, recRating)
            if userRating > 7 and recRating > 7:
                correct += 1
            elif userRating <= 4 and recRating <= 4:
                correct += 1
            elif userRating > 4 and recRating > 4 and userRating <= 7 and recRating <= 7:
                correct += 1
            elif abs(userRating - recRating) <= 1:
                correct += 1
    
    percentCorrect = correct / len(combined_ratings)
    print("[FinalProject] % Correct: " + str(percentCorrect))
    return percentCorrect, combined_ratings

# Conduct collaborative filtering algorithm to recommend movies
def collaborative_filtering(accountName):
    # get nearest neighbors 
    # (Metric: 1=manhattan, 2=euclidean, 3=cosine)
    metric = 3
    accuracyTestMode = False

    nearestNeighbors, unseen_movies, removed_movies = get_knn_collaborative_filtering(accountName, k=10, metric=metric, accuracyTestMode=accuracyTestMode)
    print("[FinalProject] Nearest Neighbors:", nearestNeighbors)

    N = 10

    recommendations = recommend_N_movies(N, nearestNeighbors, unseen_movies, metric = metric, accuracyTestMode=accuracyTestMode)

    if accuracyTestMode == True:
        print("[FinalProject] Accuracy test for collaborative filtering enabled")
        calculateAccuracyTestResults(removed_movies, recommendations)
    else:
        print("[FinalProject] Top " + str(N) + " recommended movies are: ")
        movieCount = 0
        for movie in recommendations:
            movieCount += 1
            print("#" + str(movieCount) + ": " + movie[0] + " | Score: " + str(movie[1]))

# runner
username = input("[FinalProject] Please enter your Letterboxd username: ")
selection = input("[FinalProject] Would you like to (1) get movie reccomendations through collaborative filtering or (2) train a neural network on your watched movies? ")
if (selection == "1"):
    collaborative_filtering(username)
elif (selection == "2"):
    run_nn(username)
else:
    print("[FinalProject] You did not enter (1) or (2)")
