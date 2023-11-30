# final project code goes here

from GenerateJSON import *
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


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


data, genreList = read_user_numeric("nmcassa")
genreList = [[x] for x in genreList]
le = OrdinalEncoder()
genreList = le.fit_transform(genreList)

print(genreList)

for i in range(len(data)):
    data[i][1].append(genreList[i][0])
print(data)

X = [x[1] for x in data]
sc = MinMaxScaler()
X = sc.fit_transform(X)
#np.concatenate(X,genreList)
print(X)


        