from letterboxdpy import user
from imdb import Cinemagoer
import requests
import re
import json
import copy
import time

# Spencer Rothfleisch, Louie Shapiro, Max Ward - CS 349
# get_imdb_id() function from https://github.com/TobiasPankner/Letterboxd-to-IMDb/blob/master/letterboxd2imdb.py
def get_imdb_id(letterboxd_uri):
    resp = requests.get(letterboxd_uri)
    if resp.status_code != 200:
        return None

    # extract the IMDb url
    re_match = re.findall(r'href=".+title/(tt\d+)/maindetails"', resp.text)
    if not re_match:
        return None

    return re_match[0][2:]

# collects users letterboxd reviews and converts ratings to a 1-10 scale
def getUserReviews(accountName):
    reviews = dict()
    account = user.User(accountName)
    initialReviews = user_reviews_fixed(account)
    for review in initialReviews:
        rating = review['rating'].strip()
        if rating == "½":
            reviews[review['movie']] = 1
        elif rating == "★":
            reviews[review['movie']] = 2
        elif rating == "★½":
            reviews[review['movie']] = 3
        elif rating == "★★":
            reviews[review['movie']] = 4
        elif rating == "★★½":
            reviews[review['movie']] = 5
        elif rating == "★★★":
            reviews[review['movie']] = 6
        elif rating == "★★★½":
            reviews[review['movie']] = 7
        elif rating == "★★★★":
            reviews[review['movie']] = 8
        elif rating == "★★★★½":
            reviews[review['movie']] = 9
        else:
            reviews[review['movie']] = 10
    return reviews

#gives reviews that the user selected has made, fixed by Spencer to work on every page
def user_reviews_fixed(user) -> list:
    #returns all movies
    prev = count = 0
    curr = 1
    review_list = []

    while prev != curr:
        count += 1
        print("[GenerateJSON] Parsing Review Page " + str(count) + ". Review List Length: " + str(len(review_list)))
        curr 
        prev = len(review_list)
        page = user.get_parsed_page("https://letterboxd.com/" + user.username + "/films/reviews/page/" + str(count) + "/")

        data = page.find_all("div", {"class": ["film-detail-content"], })

        for item in data:
            curr = {}

            # Ensure movie has been rated
            if item.find("span", {"class": ["rating"], }) != None:
                curr['movie'] = item.find("a").text #movie title
                curr['rating'] = item.find("span", {"class": ["rating"], }).text #movie rating
                curr['date'] = item.find("span", {"class": ["_nobr"], }).text #rating date
                curr['review'] = item.find("div", {"class": ["body-text"], }).findChildren()[0].text #review

                review_list.append(curr)

        curr = len(review_list)

    return review_list

# Gets the IMDb movie details for movies the user has reviewed on letterboxd and creates CSV details
def createUserJSON(accountName):
    start = time.time()
    print("[GenerateJSON] Creating CSV for Letterboxd account " + accountName)
    # creating an instance of the IMDB()
    ia = Cinemagoer()

    # Gets letterboxd reviews
    print("[GenerateJSON] Retrieving Reviews")
    revs = getUserReviews(accountName)

    # Gets films watched on letterboxd
    print("[GenerateJSON] Retrieving Movies Watched")
    nick = user.User(accountName)
    films_watched = user.user_films_watched(nick)

    # Creates list of movies
    movies = list(revs.keys())

    # gets dictionary of already stored movie info
    storedMovies = load_movies()

    letterboxd_uri = ""

    # iterate through all movies reviewed on the users profile
    print("[GenerateJSON] Retrieving Movie Information")
    userMovieDict = dict()
    for i in range(len(movies)):
        for n, url in films_watched:
            if n == movies[i]:
                letterboxd_url = url

        # URI for IMDb data request
        letterboxd_uri = "https://letterboxd.com/film/" + letterboxd_url + "/"

        # check if movie is already in json store, otherwise add it to json store
        if letterboxd_url in storedMovies:
            currentMovie = copy.deepcopy(storedMovies[letterboxd_url])
            currentMovie["userLetterboxdReview"] = revs[movies[i]]
            userMovieDict[letterboxd_url] = currentMovie
            print("[GenerateJSON] " + letterboxd_url + " already added to movies.json")
        else:
            try:
                imdbid = get_imdb_id(letterboxd_uri)
                imdb_movie = ia.get_movie(imdbid)
                # pick attributes to add to movie store
                movieInfo = dict()
                movieInfo["localizedTitle"] = imdb_movie.data["localized title"]
                movieInfo["topThreeActors"] = [imdb_movie.data["cast"][0].get('name', ''), imdb_movie.data["cast"][1].get('name', ''), imdb_movie.data["cast"][2].get('name', '')]
                movieInfo["genres"] = imdb_movie.data["genres"]
                movieInfo["runtimes"] = imdb_movie.data["runtimes"]
                movieInfo["colorInfo"] = imdb_movie.data["color info"]
                movieInfo["rating"] = imdb_movie.data["rating"]
                movieInfo["year"] = imdb_movie.data["year"]
                if "director" in imdb_movie.data:
                    movieInfo["director"] = imdb_movie.data["director"][0].get('name', '')
                if "writer" in imdb_movie.data:
                    movieInfo["writer"] = imdb_movie.data["writer"][0].get('name', '')
                if "producer" in imdb_movie.data:
                    movieInfo["producer"] = imdb_movie.data["producer"][0].get('name', '')
                if "composer" in imdb_movie.data:
                    movieInfo["composer"] = imdb_movie.data["composer"][0].get('name', '')

                # add movie to storedMovies dict
                storedMovies[letterboxd_url] = movieInfo
                
                # copy movie info and add it to user dict
                movieInfoUser = copy.deepcopy(movieInfo)
                movieInfoUser["userLetterboxdReview"] = revs[movies[i]]
                userMovieDict[letterboxd_url] = movieInfoUser
                save_movies(storedMovies)
                print("[GenerateJSON] Added " + letterboxd_url + " to movies.json")
            except:
                print("[GenerateJSON] Error: Could not fetch information for movie " + letterboxd_uri)

    # create userJson
    end = time.time()
    timeElapsed = (end - start) / 60
    save_user(userMovieDict, accountName)
    print("[GenerateJSON] Created " + accountName + ".json - Time elapsed: " + str(timeElapsed) + " minutes")

# Function to load data from JSON file
def load_movies():
    with open("movies.json", 'r') as file:
        return json.load(file)
        
# Function to save data to JSON file
def save_movies(data):
    with open("movies.json", 'w') as file:
        json.dump(data, file)

# Function to save user to JSON file
def save_user(data, user):
    with open("users/" + user + ".json", 'w') as file:
        json.dump(data, file)

#userToGenerate = input("Enter letterboxd username: ")
#createUserJSON(userToGenerate)