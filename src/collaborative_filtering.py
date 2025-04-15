import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from obtain_data import *

def pairwise_similarity(data_centered: pd.DataFrame, distance: str, method: str):
    '''
    Calculate pairwise similarity using the L2 norm, or Euclidean distance.
    Args:
        data_centered (pd.DataFrame): DataFrame with users as rows and items as columns.
    Returns:
        pd.DataFrame: DataFrame with item similarity scores.
    '''
    if method == 'user':
        if distance == 'euclidean':
            sim_matrix = -pairwise_distances(data_centered, metric='euclidean')
        elif distance == 'cosine':
            sim_matrix = 1 - pairwise_distances(data_centered, metric='cosine')
        np.fill_diagonal(sim_matrix, 1)
        return pd.DataFrame(sim_matrix, index=data_centered.T.columns, columns=data_centered.T.columns)
    elif method == 'item':
        if distance == 'euclidean':
            sim_matrix = -pairwise_distances(data_centered.T, metric='euclidean')
        elif distance == 'cosine':
            sim_matrix = 1 - pairwise_distances(data_centered.T, metric='cosine')
        np.fill_diagonal(sim_matrix, 1)
        return pd.DataFrame(sim_matrix, index=data_centered.columns, columns=data_centered.columns)

def scale_data(data: pd.DataFrame):
    '''
    Scales the input DataFrame using MinMaxScaler.
    Args:
        data (pd.DataFrame): DataFrame with users as rows and items as columns.
    Returns:
        pd.DataFrame: Scaled DataFrame.
    '''
    data.columns = data.columns.astype(str)
    data.index = data.index.astype(str)

    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_array, index=data.index, columns=data.columns)
    return scaled_data

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
        pd.Series: Predicted ratings for the target user. Includes all items for use in evaluation.
    '''
    if target_user not in data.index:
        return f"Error: {target_user} not found in the dataset."

    if data.loc[target_user].isnull().sum() == 0:
        return f"{target_user} has no missing ratings."
    
    data_centered = data.sub(data.mean(axis=0), axis=1).fillna(0)

    sim_df = pairwise_similarity(data_centered, distance, method='item')
    k = min(k, len(sim_df) - 1)

    scaled_sim_df = sim_df.copy()
    if scale:
        scaled_sim_df = scale_data(scaled_sim_df)

    target_ratings = data.loc[target_user]
    preds = {}

    for item in data.columns:
        raters = data[item].dropna().index.difference([target_user])
        if len(raters) == 1:
            preds[item] = data[item].mean()
        else:
            item_similarities = scaled_sim_df[item].drop(index=item).dropna()
            top_k_similar = item_similarities.nlargest(k)
            if top_k_similar.empty or top_k_similar.sum() == 0:
                preds[item] = data[item].mean()
            else:
                rated_similar_items = target_ratings[top_k_similar.index].dropna()
                if not rated_similar_items.empty:
                    relevant_similarities = top_k_similar[rated_similar_items.index]
                    weighted_sum = np.dot(rated_similar_items, relevant_similarities)
                    sum_of_weights = relevant_similarities.sum()

                    preds[item] = weighted_sum / sum_of_weights if sum_of_weights != 0 else data[item].mean()
                else:
                    preds[item] = data[item].mean()

    return pd.Series(preds, index=data.columns)

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
        pd.Series: Predicted ratings for the target user. Includes all items for use in evaluation.
    '''
    if target_user not in data.index:
        return f"Error: {target_user} not found in the dataset."

    if data.loc[target_user].isnull().sum() == 0:
        return f"{target_user} has no missing ratings."

    data_centered = data.sub(data.mean(axis=1), axis=0).fillna(0)
    sim_df = pairwise_similarity(data_centered, distance, method='user')
    k = min(k, len(sim_df) - 1)

    scaled_sim_df = sim_df.copy()
    if scale:
        scaled_sim_df = scale_data(scaled_sim_df)

    preds = {}
    for item in data.columns:
        similar_users = scaled_sim_df.loc[target_user].drop(index=target_user).dropna()
        relevant_ratings = data.loc[similar_users.index, item].dropna()
        
        if relevant_ratings.empty:
            preds[item] = data[item].mean()
            continue

        top_k_similar = similar_users[relevant_ratings.index].nlargest(k)

        if top_k_similar.empty or top_k_similar.sum() == 0:
            preds[item] = data[item].mean()
        else:
            neighbor_ratings = data.loc[top_k_similar.index, item]
            similarity_weights = top_k_similar
            preds[item] = np.dot(neighbor_ratings, similarity_weights) / similarity_weights.sum()

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

def root_mean_squared_error(y_true, y_pred):
    '''
    Calculate the root mean squared error between true and predicted ratings.
    Args:
        y_true (pd.Series): True ratings.
        y_pred (pd.Series): Predicted ratings.
    Returns:
        float: RMSE value.
    '''
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true, y_pred):
    '''
    Calculate the mean absolute error between true and predicted ratings.
    Args:
        y_true (pd.Series): True ratings.
        y_pred (pd.Series): Predicted ratings.
    Returns:
        float: MAE value.
    '''
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(data: pd.DataFrame, target_user: str, k: int, distance: str, method: str, scale: bool = False):
    '''
    Test the collaborative filtering method.
    Args:
        data (pd.DataFrame): DataFrame with users as rows and items as columns.
        target_user (str): The user for whom we want to predict ratings.
        k (int): The number of similar items/users to consider.
        distance (str): The distance metric to use ('euclidean' or 'cosine').
        scale (bool): Whether to scale the data before calculating similarity.
    Returns:
        None
    '''
    if method == 'user':
        preds = collab_filter_user(data, target_user, k, distance, scale)
    elif method == 'item':
        preds = collab_filter_item(data, target_user, k, distance, scale)
    else:
        raise ValueError("Invalid method. Choose 'user' or 'item'.")
    print(f"Predicted ratings for {target_user}, method={method}, k={k}, distance={distance}, scale={scale}")
    true_ratings = data.loc[target_user].dropna()
    common_items = true_ratings.index.intersection(preds.index)

    if common_items.empty:
        print("No common items. Cannot calculate RMSE and MAE.")
        return np.nan, np.nan
    else:
        true_ratings_common = true_ratings.loc[common_items]
        preds_common = preds.loc[common_items]

        rmse = root_mean_squared_error(true_ratings_common, preds_common)
        mae = mean_absolute_error(true_ratings_common, preds_common)

        print(f"RMSE: {rmse}, MAE: {mae}")
        return rmse, mae
