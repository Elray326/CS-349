from letterboxdpy import user
import imdb
import requests
import re

# creating an instance of the IMDB()
ia = imdb.IMDb()


# get_imdb_id() function from https://github.com/TobiasPankner/Letterboxd-to-IMDb/blob/master/letterboxd2imdb.py
def get_imdb_id(letterboxd_uri):
    resp = requests.get(letterboxd_uri)
    if resp.status_code != 200:
        return None

    # extract the IMDb url
    re_match = re.findall(r'href=".+title/(tt\d+)/maindetails"', resp.text)
    if not re_match:
        return None

    return re_match[0]


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

print(get_imdb_id(letterboxd_uri))