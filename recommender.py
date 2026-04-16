import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st


# Columns to display
USE_COLS = ['title', 'type', 'listed_in', 'country', 'release_year','combined_features']

@st.cache_resource
def load_model():

    data_model = pd.read_csv("cleaned_netflix_data.csv")

    # Create combined features
    data_model['combined_features'] = (
        data_model['listed_in'].fillna('') + " " +
        data_model['description'].fillna('')
    )

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_model['combined_features'])

    # Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Title index mapping
    indices = pd.Series(
        data_model.index,
        index=data_model['title']
    ).drop_duplicates()

    return data_model, cosine_sim, indices



# Recommendation logic

def recommend(query, top_n=10):

    # Load cached data
    data_model, cosine_sim, indices = load_model()

    query = query.lower().strip()

    
    # CASE 1: Title match
    
    title_match = indices.index.str.lower()

    if query in title_match.values:

        idx = indices[title_match == query].values[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip itself
        sim_scores = sim_scores[1: top_n + 1]

        movie_indices = [i[0] for i in sim_scores]

        selected = data_model.loc[[idx], USE_COLS]
        recommendations = data_model.loc[movie_indices, USE_COLS]

        return selected, recommendations

    
    # CASE 2: Genre search
    
    genre_results = data_model[
        data_model['listed_in'].str.lower().str.contains(query, na=False)
    ]

    if genre_results.empty:
        return None, None

    selected = genre_results.iloc[[0]][USE_COLS]
    recommendations = genre_results.iloc[1: top_n + 1][USE_COLS]

    return selected, recommendations