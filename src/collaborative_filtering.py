import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from obtain_data import *

def scale_data(data: pd.DataFrame):
    '''
    Scales the input DataFrame using MinMaxScaler.
    Args:
        data (pd.DataFrame): DataFrame with users as rows and items as columns.
    Returns:
        pd.DataFrame: Scaled DataFrame.
    '''
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

def pairwise_item_similarity(data_centered: pd.DataFrame, distance: str):
    '''
    Calculate pairwise item similarity using the L2 norm, or Euclidean distance.
    Args:
        data_centered (pd.DataFrame): DataFrame with users as rows and items as columns.
    Returns:
        pd.DataFrame: DataFrame with item similarity scores.
    '''
    if distance == 'euclidean':
        sim_matrix = -pairwise_distances(data_centered.T, metric='euclidean')
    elif distance == 'cosine':
        sim_matrix = 1 - pairwise_distances(data_centered.T, metric='cosine')

    np.fill_diagonal(sim_matrix, 1)
    return pd.DataFrame(sim_matrix, index=data_centered.columns, columns=data_centered.columns)

def collab_filter_item(data: pd.DataFrame, target_user: str, k: int, distance: str, scale: bool = False):
    '''
    Collaborative filtering using item-based approach.
    Args:
        data (pd.DataFrame): DataFrame with users as rows and items as columns.
        target_user (str): The user for whom we want to predict ratings.
        k (int): The number of similar items to consider.
        distance (str): The distance metric to use ('euclidean' or 'cosine').
        scale (bool): Whether to scale the data before calculating similarity.
    Returns:
        pd.Series: Predicted ratings for the target user.
    '''
    if target_user not in data.index:
        return f"Error: {target_user} not found in the dataset."

    if data.loc[target_user].isnull().sum() == 0:
        return f"{target_user} has no missing ratings."

    scaled_data = data.copy()
    if scale:
        scaled_data = scale_data(scaled_data)

    data_centered = scaled_data.sub(scaled_data.mean(axis=0), axis=1).fillna(0)

    sim_df = pairwise_item_similarity(data_centered, distance)
    k = min(k, len(sim_df))

    target_ratings = data.loc[target_user]
    missing_items = target_ratings[target_ratings.isna()].index

    preds = {}
    for item in missing_items:
        similar_items = sim_df[item].drop(index=item).dropna()
        top_k_similar = similar_items.nlargest(min(k, len(similar_items)))

        if top_k_similar.empty or top_k_similar.sum() == 0:
            preds[item] = data[item].mean()
        else:
            filled_ratings = data[top_k_similar.index].loc[target_user].fillna(data[top_k_similar.index].mean())
            preds[item] = np.dot(filled_ratings, top_k_similar) / top_k_similar.sum()

    return pd.Series(preds)


def collab_filter_user(data: pd.DataFrame, target_user: str, k: int, distance: str, scale: bool = False):
    '''
    Collaborative filtering using user-based approach.
    Args:
        data (pd.DataFrame): DataFrame with users as rows and items as columns.
        target_user (str): The user for whom we want to predict ratings.
        k (int): The number of similar users to consider.
        distance (str): The distance metric to use ('euclidean' or 'cosine').
        scale (bool): Whether to scale the data before calculating similarity.
    Returns:
        pd.Series: Predicted ratings for the target user.
    '''
    if target_user not in data.index:
        return f"Error: {target_user} not found in the dataset."

    if data.loc[target_user].isnull().sum() == 0:
        return f"{target_user} has no missing ratings."

    scaled_data = data.copy()
    if scale:
        scaled_data = scale_data(scaled_data)

    data_centered = scaled_data.sub(scaled_data.mean(axis=1), axis=0).fillna(0)
    
    sim_df = pairwise_item_similarity(data_centered, distance)
    k = min(k, len(sim_df))

    target_ratings = data.loc[target_user]
    missing_items = target_ratings[target_ratings.isna()].index

    preds = {}
    for item in missing_items:
        similar_users = sim_df[item].drop(index=item).dropna()
        top_k_similar = similar_users.nlargest(min(k, len(similar_users)))

        if top_k_similar.empty or top_k_similar.sum() == 0:
            preds[item] = data[item].mean()
        else:
            filled_ratings = data[top_k_similar.index].loc[target_user].fillna(data[top_k_similar.index].mean())
            preds[item] = np.dot(filled_ratings, top_k_similar) / top_k_similar.sum()

    return pd.Series(preds)

def recommend_movies(preds: pd.Series, n_recommendations: int = 10, min_watchers: int = 1):
    '''
    Recommend movies based on predicted ratings.
    Args:
        preds (pd.Series): Predicted ratings for movies.
        n_recommendations (int): Number of recommendations to return.
    Returns:
        pd.DataFrame: DataFrame containing recommended movies and their details.
    '''
    recommend_movies_df = preds.nlargest(n_recommendations * 3).sample(n_recommendations * 3, random_state=1).to_frame(name='predicted_rating')
    recommend_movies_df.index.name = 'ids.trakt'
    recommend_movies_df = recommend_movies_df.reset_index()

    for index, row in recommend_movies_df.iterrows():
        movie_id = row['ids.trakt']
        url = f'{BASE_URL}/movies/{movie_id}?extended=full,images'
        response = request_trakt(url)
        if response:
            data = json.loads(response)
            title = data.get('title', None)
            year = data.get('year', None)
            certification = data.get('certification', None)
            poster = data.get('images', {}).get('poster', [])

            recommend_movies_df.at[index, 'title'] = title
            recommend_movies_df.at[index, 'year'] = year
            recommend_movies_df.at[index, 'certification'] = certification
            recommend_movies_df.at[index, 'poster_url'] = f'https://{poster[0]}' if len(poster) > 0 else None
        url = f'{BASE_URL}/movies/{movie_id}/watching'
        response = request_trakt(url)
        if response:
            data = json.loads(response)
            watchers = len(data)
            recommend_movies_df.at[index, 'watchers'] = watchers
        else:
            recommend_movies_df.at[index, 'watchers'] = 0

    recommend_movies_df = recommend_movies_df[recommend_movies_df['watchers'] >= min_watchers].head(n_recommendations)

    return recommend_movies_df
