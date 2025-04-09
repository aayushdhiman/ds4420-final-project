import streamlit as st
from collaborative_filtering import *
import pandas as pd

from PIL import Image
import os
import base64
import shutil
import time

def copy_image_to_clicked_folder(image_path, trakt_id):
    if os.path.exists('clicked'):
        shutil.rmtree('clicked')
    os.makedirs('clicked', exist_ok=True)
    clicked_image_path = os.path.join('clicked', f"{trakt_id}.jpg")
    shutil.copy(image_path, clicked_image_path)

def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

def load_images(trending_df, image_folder):
    if not os.path.exists(image_folder):
        st.error(f"Error: The folder '{image_folder}' does not exist.")
        return []
    
    else:
        num_cols = 10
        cols = st.columns(num_cols)
        
        for index, row in trending_df.iterrows():
            trakt_id = row['ids.trakt']
            title = row['title']
            try:
                year = int(row['year'])
            except ValueError:
                year = row['year']
            
            image_path = os.path.join(image_folder, f"{trakt_id}.jpg")
            if os.path.exists(image_path):
                img_base64 = get_image_as_base64(image_path)
                button_html = f'''
                    <button style="border:none; background:none; padding:0;">
                        <img src="data:image/jpeg;base64,{img_base64}" 
                            style="width:100%; height:auto; border-radius:8px;" />
                    </button>
                '''
                with cols[index % num_cols]:
                    st.markdown(button_html, unsafe_allow_html=True)
                    if st.button(f"{title} ({year})", key=f"btn_{index}", on_click=lambda t=title, y=year, tid=trakt_id, ip=image_path: (
                        st.session_state.update({"selected_movie": {"trakt_id": tid, "title": t, "year": y}}),
                        copy_image_to_clicked_folder(ip, tid),
                        st.experimental_set_query_params(page="Collab Filtering")
                    )):
                        pass

def display_clicked(selected_movie):
    '''
    Display the clicked movie image and information.
    Args:
        selected_movie (dict): Dictionary containing movie information (trakt_id, title, year).
    '''
    trakt_id = selected_movie.get('trakt_id')
    title = selected_movie.get('title')
    year = selected_movie.get('year')
    clicked_image_path = os.path.join('clicked', f"{trakt_id}.jpg")
    if os.path.exists(clicked_image_path):
        img_base64 = get_image_as_base64(clicked_image_path)
        img_html = f'''
            <img src="data:image/jpeg;base64,{img_base64}"
                 style="width:30%; height:auto; border-radius:8px; margin-bottom: 10px;" />
            <p style="text-align:center;">{title} ({year})</p>
        '''
        st.markdown(img_html, unsafe_allow_html=True)
    else:
        st.warning("Could not load the selected movie image.")

def display_recommended(num_cols, n_recommendations, recommend_movies_df, image_folder):
    '''
    Display images of recommended movies.
    Args:
        n_recommendations (int): Number of recommendations to display.
        recommend_movies_df (pd.DataFrame): DataFrame containing recommended movies.
        image_folder (str): Path to the folder containing movie images.
    '''
    cols = st.columns(num_cols)
    image_index = 0
    if n_recommendations < len(recommend_movies_df):
        recommend_movies_df = recommend_movies_df.sample(n_recommendations, random_state=1)
    else:
        recommend_movies_df = recommend_movies_df.nlargest(n_recommendations, 'predicted_rating')
    recommend_movies_df = recommend_movies_df.reset_index(drop=True)
    for index, row in recommend_movies_df.iterrows():
        trakt_id = row['ids.trakt']
        title = row['title']
        rating = row['predicted_rating']
        watchers = row['watchers']
        try:
            year = int(row['year'])
        except ValueError:
            year = row['year']
        image_path = os.path.join(image_folder, f"{trakt_id}.jpg")
        if os.path.exists(image_path):
            img_base64 = get_image_as_base64(image_path)
            img_html = f'''
                <img src="data:image/jpeg;base64,{img_base64}"
                     style="width:80%; height:auto; border-radius:8px; margin-bottom: 10px;" />
                <p style="text-align:center;">{title} ({year})</p>
                <p style="text-align:center;">Predicted Rating: {rating}</p>
                <p style="text-align:center;">Current Watchers: {watchers}</p>
            '''
            with cols[image_index % num_cols]:
                st.markdown(img_html, unsafe_allow_html=True)
            image_index += 1
    return None

st.set_page_config(page_title="Movie Calculator", page_icon=":movie_camera:", layout="wide")

if "selected_movie" not in st.session_state:
    st.session_state["selected_movie"] = {}

# Get the page from query parameters, default to "Home"
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["Home"])[0]

# page = st.radio("Choose a page:", ["Home", "Collab Filtering", "CNN"])

st.markdown("<h1 style='text-align: center; color: red; '>Movie Calculator</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: red; '>Dhruv Gandhi and Aayush Dhiman</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: tan; '>DS 4420 Final Project:</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: tan; '>Collaborative Filtering and Convolutional Neural Network</h3>", unsafe_allow_html=True)

st.markdown("\n\n\n\n", unsafe_allow_html=True)
image_folder = "movie_posters"
trending_df = pd.read_csv("movie_info/trakt_movie_info.csv")

if page == "Home":
    st.markdown("<h6 style='text-align: center; color: tan; '>Select a movie you've watched and rate it out of 10</h6>", unsafe_allow_html=True)
    load_images(trending_df, image_folder)
elif page == "Collab Filtering":
    st.write("This is the Collaborative Filtering page.")
    if st.session_state.get('selected_movie'):
        selected_movie = st.session_state['selected_movie']
        trakt_id = selected_movie.get('trakt_id')
        title = selected_movie.get('title')
        year = selected_movie.get('year')

        st.subheader(f"You selected:")
        display_clicked(selected_movie)

        st.write(f"**How would you rate _{title}_?** 🎬")
        rating = st.slider("Rate this movie:", 0, 10)
        st.write(f"You rated _{title}_ a {rating}/10!")
        if rating > 0:
            user_entry = create_user_entry(rating, trakt_id)
            st.write("Generating recommendations...")
            if st.button("L2 Norm Item-Item"):
                st.session_state['distance'] = 'euclidean'
                st.session_state['collab_filter'] = 'item_item'
                st.write("Using L2 Norm (Euclidean distance) for item-item collaborative filtering.")
            elif st.button("Cosine Similarity Item-Item"):
                st.session_state['distance'] = 'cosine'
                st.session_state['collab_filter'] = 'item_item'
                st.write("Using Cosine Similarity for item-item collaborative filtering.")
            elif st.button("L2 Norm User-User"):
                st.session_state['distance'] = 'euclidean'
                st.session_state['collab_filter'] = 'user_user'
                st.write("Using L2 Norm (Euclidean distance) for user-user collaborative filtering.")
            elif st.button("Cosine Similarity User-User"):
                st.session_state['distance'] = 'cosine'
                st.session_state['collab_filter'] = 'user_user'
                st.write("Using Cosine Similarity for user-user collaborative filtering.")
            start = time.time()
            if st.session_state.get('collab_filter') == 'item_item':
                if st.session_state.get('distance') == 'euclidean':
                    ratings = collab_filter_item(user_entry, 'temp_user_1', 5, distance='euclidean')
                else:
                    ratings = collab_filter_item(user_entry, 'temp_user_1', 5, distance='cosine')
            else:
                if st.session_state.get('distance') == 'euclidean':
                    ratings = collab_filter_user(user_entry, 'temp_user_1', 5, distance='euclidean')
                else:
                    ratings = collab_filter_user(user_entry, 'temp_user_1', 5, distance='cosine')
            st.write(f"Recommendations generated in {time.time() - start:.2f} seconds.")
            recs = store_movie_posters('recommendations', recommend_movies(ratings, 10, 0))
            st.subheader("Here are the Recommended Movies:")
            if recs is not None and not recs.empty:
                display_recommended(5, 10, recs, 'recommendations')
            else:
                st.info("No recommendations available yet.")

    else:
        st.info("Please select a movie from the Home page to rate.")
elif page == "CNN":
    st.write("This is the CNN page.")

