import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from recommender import recommend


# Page Config

st.set_page_config(page_title="Netflix Dashboard", layout="wide")
st.title("Netflix Content Analysis Dashboard")


# Load Data

data = pd.read_csv("netflix_titles.csv")


# Data Cleaning

data = data[~data['rating'].str.contains('min', na=False)]

data['decade'] = (data['release_year'] // 10) * 10
data['content_age'] = 2024 - data['release_year']
data['duration_num'] = data['duration'].str.extract(r'(\d+)').astype(float)


# Sidebar Filters

st.sidebar.header("Filters")

# Content Type
type_filter = st.sidebar.selectbox(
    "Content Type",
    ["All"] + list(data['type'].dropna().unique())
)

# Country
country_filter = st.sidebar.selectbox(
    "Country",
    ["All"] + list(data['country'].dropna().unique())
)

# Year Slider (dynamic)
year_filter = st.sidebar.slider(
    "Release Year",
    int(data['release_year'].min()),
    int(data['release_year'].max()),
    (int(data['release_year'].min()), int(data['release_year'].max()))
)

# Genre Filter
genres = data['listed_in'].str.split(',').explode().str.strip()
genre_filter = st.sidebar.selectbox(
    "Genre",
    ["All"] + list(genres.dropna().unique())
)


# Apply Filters

filtered_data = data.copy()

if type_filter != "All":
    filtered_data = filtered_data[filtered_data['type'] == type_filter]

if country_filter != "All":
    filtered_data = filtered_data[filtered_data['country'] == country_filter]

filtered_data = filtered_data[
    (filtered_data['release_year'] >= year_filter[0]) &
    (filtered_data['release_year'] <= year_filter[1])
]

if genre_filter != "All":
    filtered_data = filtered_data[
        filtered_data['listed_in'].str.contains(genre_filter, na=False)
    ]


# KPIs

col1, col2, col3 = st.columns(3)

col1.metric("Total Titles", len(filtered_data))
col2.metric("Movies", len(filtered_data[filtered_data['type'] == 'Movie']))
col3.metric("TV Shows", len(filtered_data[filtered_data['type'] == 'TV Show']))

st.divider()


# Movies vs TV + Growth

col4, col5 = st.columns(2)

with col4:
    fig1 = px.histogram(filtered_data, x="type", title="Movies vs TV Shows")
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    decade = filtered_data['decade'].value_counts().sort_index()
    fig2 = px.line(decade, title="Content Growth Over Decades")
    st.plotly_chart(fig2, use_container_width=True)


# Top Genres + Countries

col6, col7 = st.columns(2)

with col6:
    genres = filtered_data['listed_in'].str.split(',').explode().str.strip()
    top_genres = genres.value_counts().head(10).sort_values()

    fig3 = px.bar(top_genres, orientation='h', title="Top Genres")
    st.plotly_chart(fig3, use_container_width=True)

with col7:
    top_countries = filtered_data['country'].value_counts().head(10).sort_values()

    fig4 = px.bar(top_countries, orientation='h', title="Top Countries")
    st.plotly_chart(fig4, use_container_width=True)

# Top Genres Per Year

genre_year = filtered_data.copy()
genre_year['listed_in'] = genre_year['listed_in'].str.split(',')

genre_year = genre_year.explode('listed_in')
genre_year['listed_in'] = genre_year['listed_in'].str.strip()

top_genre_year = (
    genre_year.groupby(['release_year', 'listed_in'])
    .size()
    .reset_index(name='count')
)

fig5 = px.line(
    top_genre_year,
    x='release_year',
    y='count',
    color='listed_in',
    title="Top Genres Per Year"
)

st.plotly_chart(fig5, use_container_width=True)


# Rating Analysis

fig6 = px.histogram(
    filtered_data,
    x="rating",
    color="type",
    barmode="group",
    title="Rating Distribution"
)

st.plotly_chart(fig6, use_container_width=True)


# Clustering (KMeans)

import joblib
import pandas as pd

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

data_model = pd.read_csv("final_clustered_data.csv")

import plotly.express as px

fig = px.scatter(
    data_model,
    x="duration_clean",
    y="content_age",
    color="cluster",
    title="Content Clusters"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### 🔍 Cluster Insights")

cluster_summary = data_model.groupby("cluster")[[
    "duration_clean", "content_age"
]].mean()

st.dataframe(cluster_summary)

st.markdown("""
### 🔍 Cluster Insights
- Cluster 0: Short content, newer
- Cluster 1: Medium duration
- Cluster 2: Longer content
- Cluster 3: Old + long content
""")

col8, col9 = st.columns(2)

with col8:
    import plotly.express as px
    cluster_profile = data_model.groupby("cluster")[[
    "rating_enc", "duration_clean", "content_age"
    ]].mean().reset_index()

    fig1 = px.bar(
        cluster_profile,
        x="cluster",
        y=["rating_enc", "duration_clean", "content_age"],
        barmode="group",
        title="Cluster Characteristics"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col9:
    cluster_type = data_model.groupby(["cluster", "type"]).size().reset_index(name="count")

    fig2 = px.bar(cluster_type,
        x="cluster",
        y="count",
        color="type",
        barmode="group",
        title="Cluster vs Content Type Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("## 🎬 OTT Content Strategy Tool")

query = st.text_input("Enter Movie Title or Genre", key="recommend_input")

if st.button("Recommend"):
    st.subheader(f"Results for: {query}")

    selected, recs = recommend(query)

    if selected is not None:
        st.subheader("🎯 Selected Content")
        st.dataframe(selected)

        st.subheader("🔥 Recommended Content")
        st.dataframe(recs)
    else:
        st.error("No results found")




