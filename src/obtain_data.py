
import os
import pandas as pd
from urllib.request import Request, urlopen
import json
import time
import numpy as np

client_id = os.environ.get('client_id')
client_secret = os.environ.get('client_secret')
BASE_URL = "https://api.trakt.tv"

def request_trakt(url):
    '''
    Makes a request to the Trakt API and handles rate limiting by sleeping every 5 seconds.
    Args:
        url (str): The URL to request.
    Returns:
        response_body (str): The response body from the API.
    '''
    headers = {
        "trakt-api-version": "2",
        "Content-Type": "application/json",
        "trakt-api-key": client_id
    }
    request = Request(url, headers=headers)
    try:
        response_body = urlopen(request).read()
        return response_body
    except Exception as e:
        if hasattr(e, 'code') and e.code == 429:
            time.sleep(5)
            return request_trakt(url)
        else:
            # print("Error: ", e)
            return None

def get_trending_movies(pages=15):
    '''
    Fetches trending movies from the Trakt API.
    Args:
        pages (int): The number of pages to fetch.
    Returns:
        trending_df (DataFrame): A DataFrame containing the trending movies.
    '''
    trending_df = pd.DataFrame()

    for page in range(1, pages+1):
        url = f"{BASE_URL}/movies/trending?page={page}"
        response = request_trakt(url)

        if response:
            data = json.loads(response)
            trending_df = pd.concat([trending_df, pd.DataFrame(data)], ignore_index=True)
    
    watchers = trending_df['watchers']
    trending_df = pd.json_normalize(trending_df['movie'])
    trending_df['watchers'] = watchers
    return trending_df

def get_usernames_of_watchers(movie_ids):
    '''
    Fetches usernames of users currently watching the specified movies.
    Args:
        movie_ids (list): A list of movie IDs.
    Returns:
        username_set (set): A set of usernames.
    '''
    username_set = set()
    for i in range(len(movie_ids)):
        url = f"{BASE_URL}/movies/{movie_ids[i]}/watching"
        response = request_trakt(url)
        if response:
            data = json.loads(response)
            watching_movie_df = pd.DataFrame(data)
            for i in range(len(watching_movie_df)):
                formatted_username = watching_movie_df.loc[i]['username'].replace(' ', '-')
                username_set.add(formatted_username)

    return username_set

def get_movie_ratings(username_set):
    '''
    Fetches all movie ratings for the specified users.
    Args:
        username_set (set): A set of usernames.
    Returns:
        overall_movie_ratings_df (DataFrame): A DataFrame containing the movie ratings.
    '''
    overall_movie_ratings_df = pd.DataFrame(columns=['username', 'movie_id', 'rating'])

    for username in username_set:
        url = f"{BASE_URL}/users/{username}/ratings"
        response = request_trakt(url)
        if response:
            data = json.loads(response)
            ratings_df = pd.DataFrame(data)
            if len(ratings_df) > 0:
                if 'type' in ratings_df.columns:
                    ratings_movie_df = ratings_df[ratings_df['type'] == 'movie']

                    if len(ratings_movie_df) > 0:
                        ratings_movie_df = ratings_movie_df[['rating', 'movie']].reset_index(drop=True)

                    user_df = pd.DataFrame()
                    for i in range(len(ratings_movie_df)):
                        movie_info = ratings_movie_df.loc[i]['movie']
                        movie_id = movie_info['ids']['trakt']
                        rating = ratings_movie_df.loc[i]['rating']
                        
                        user_df = pd.concat([user_df, pd.DataFrame({'username': [username], 
                                                                    'movie_id': [movie_id], 
                                                                    'rating': [rating]})], ignore_index=True)
                    overall_movie_ratings_df = pd.concat([overall_movie_ratings_df, user_df], ignore_index=True)

    overall_movie_ratings_df = overall_movie_ratings_df.drop_duplicates(subset=['username', 'movie_id'])
    pivoted = overall_movie_ratings_df.pivot(index='username', columns='movie_id', values='rating')
    return pivoted

def get_movie_certifications_and_poster_urls(trending_df):
    '''
    Fetches the certifications and poster URLs for a list of movie IDs from the Trakt API.
    Args:
        trending_df (DataFrame): A DataFrame containing movie IDs.
    Returns:
        trending_df (DataFrame): The same DataFrame with additional columns for certifications and poster URLs.
    '''

    trending_df['certification'] = None
    trending_df['poster_url'] = None
    for index, row in trending_df.iterrows():
        movie_id = row['ids.trakt']
        url = f'{BASE_URL}/movies/{movie_id}?extended=full,images'
        response = request_trakt(url)
        if response:
            data = json.loads(response)
            cert = data.get('certification', None)
            poster = data.get('images', {}).get('poster', [])
            
            if cert is not None:
                trending_df.at[index, 'certification'] = cert
            else:
                trending_df.at[index, 'certification'] = 'N/A'
            
            if len(poster) > 0:
                trending_df.at[index, 'poster_url'] = f'https://{poster[0]}'
            else:
                trending_df.at[index, 'poster_url'] = 'N/A'
    return trending_df

def store_movie_posters(filepath, trending_df):
    '''
    Downloads and stores movie posters in the specified directory. 
    Trakt API requires all images to be cached and not hotlinked.
    Args:
        filepath (str): The directory to store the posters.
        trending_df (DataFrame): A DataFrame containing the movie IDs and their corresponding poster URLs.
    '''
    trending_df['filepath'] = None
    if os.path.exists(filepath) is False:
        os.makedirs(filepath)
    for index, row in trending_df.iterrows():
        movie_id = row['ids.trakt']
        poster_url = row['poster_url']
        poster_filename = os.path.join(filepath, f"{movie_id}.jpg")
        if not os.path.exists(poster_filename):
            try:
                req = Request(poster_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req) as response, open(poster_filename, 'wb') as out_file:
                    out_file.write(response.read())
                trending_df.at[index, 'filepath'] = poster_filename
            except Exception as e:
                # print(f"Error downloading {poster_url}: {e}")
                trending_df.at[index, 'filepath'] = np.NaN
        else:
            # print(f"File {poster_filename} already exists, skipping download.")
            trending_df.at[index, 'filepath'] = poster_filename
    return trending_df

def create_user_entry(rating, movie_id):
    '''
    Creates a user entry for the given rating and movie ID.
    '''
    user_entry = pd.DataFrame({'username': ['temp_user_1'], movie_id: [rating]})
    user_entry = user_entry.set_index('username')
    pivoted = pd.read_csv('movie_info/trakt_movie_ratings.csv')
    pivoted = pivoted.set_index('username')

    pivoted = pd.concat([pivoted, user_entry], ignore_index=False)
    
    return pivoted

def main(pages=15, ratings=True):
    start = time.time()
    trending_df = get_trending_movies(pages)
    print(f'Fetched {len(trending_df)} trending movies in {(time.time() - start):.2f} seconds.')

    trending_df = get_movie_certifications_and_poster_urls(trending_df)
    trending_df.to_csv('movie_info/trakt_movie_info.csv', index=False)
    print(f'Movie certifications and poster URLs fetched/stored successfully in {(time.time() - start):.2f} seconds.')

    store_movie_posters('movie_posters', trending_df)
    print(f'Movie posters stored successfully in {(time.time() - start):.2f} seconds.')

    if ratings:
        movie_ids = list(trending_df['ids.trakt'])
        username_set = get_usernames_of_watchers(movie_ids)
        print(f'Fetched usernames of watchers for {len(movie_ids)} movies in {(time.time() - start):.2f} seconds.')

        overall_movie_ratings_df = get_movie_ratings(username_set)
        overall_movie_ratings_df.to_csv('movie_info/trakt_movie_ratings.csv')
        print(f'Fetched movie ratings for {len(username_set)} users in {(time.time() - start):.2f} seconds.')
if __name__ == "__main__":
    main()
