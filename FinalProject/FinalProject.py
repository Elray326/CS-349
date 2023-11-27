# final project code goes here

from GenerateJSON import *
import os

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

print(read_user("schaffrillas"))
        