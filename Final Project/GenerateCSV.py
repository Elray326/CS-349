from letterboxdpy import user
import imdb
import requests
import re

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
    initialReviews = user.user_reviews(account)
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



# creating an instance of the IMDB()
ia = imdb.IMDb()

accountName = "nmcassa"
revs = getUserReviews(accountName)
nick = user.User(accountName)

films_watched = user.user_films_watched(nick)

movies = list(revs.keys())

letterboxd_uri = ""

for n, url in films_watched:
    if n == movies[0]:
        #print(n,url)
        letterboxd_url = url

letterboxd_uri = "https://letterboxd.com/film/" + letterboxd_url + "/"

imdbid = get_imdb_id(letterboxd_uri)

imdb_movie = ia.get_movie(imdbid)



# this is the list of all the attributes that we can get directly from the IMDbPY library
print(list(imdb_movie.data.keys()))
# ['localized title', 'cast', 'genres', 'runtimes', 'countries', 'country codes', 'language codes', 'color info', 'aspect ratio', 'sound mix', 'certificates', 'original air date', 'rating', 'votes', 'cover url', 'imdbID', 'videos', 'plot outline', 'languages', 'title', 'year', 'kind', 'original title', 'director', 'writer', 'producer', 'composer', 'cinematographer', 'editor', 'casting director', 'production design', 'costume designer', 'make up', 'assistant director', 'art department', 'sound crew', 'special effects', 'visual effects', 'stunt performer', 'camera and electrical department', 'casting department', 'costume department', 'location management', 'transportation department', 'miscellaneous crew', 'akas', 'production companies', 'distributors', 'special effects companies', 'other companies', 'plot', 'synopsis']

# for example
print(imdb_movie["genres"])