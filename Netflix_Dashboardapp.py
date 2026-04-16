import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from recommender import recommend


# Page Config

st.set_page_config(page_title="Netflix Dashboard", layout="wide",initial_sidebar_state="expanded")


st.markdown("""
    <style>
    .stApp { background-color: #141414; }
    section[data-testid="stSidebar"] { background-color: #000000; }
    div[data-testid="stMetric"] {
        background-color: #222222;
        border-left: 5px solid #E50914;
        padding: 15px;
        border-radius: 5px;
    }
    h1, h2, h3 { color: #E50914 !important; }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 4px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
    <h1 style="color: #E50914; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 3rem; font-weight: bold; margin-bottom: 20px;">
        NETFLIX CONTENT ANALYSIS DASHBOARD
    </h1>
    """, unsafe_allow_html=True)


# Load Data

data = pd.read_csv("netflix_titles.csv")


# Data Cleaning

data = data[~data['rating'].str.contains('min', na=False)]

data['decade'] = (data['release_year'] // 10) * 10
data['content_age'] = 2024 - data['release_year']
data['duration_num'] = data['duration'].str.extract(r'(\d+)').astype(float)


# Sidebar Filters
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", 
    width=150
)
st.sidebar.markdown("---") # Add a divider line below the logo
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
    fig1 = px.histogram(filtered_data, 
            x="type", 
            title="Movies vs TV Shows",
            color_discrete_sequence=['#E50914'], # Netflix Red
            template="plotly_dark")
    
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    decade = filtered_data['decade'].value_counts().sort_index()
    fig2 = px.line(decade, title="Content Growth Over Decades",
                   color_discrete_sequence=['#E50914'], # Netflix Red
                   template="plotly_dark")
    
    st.plotly_chart(fig2, use_container_width=True)


# Top Genres + Countries

col6, col7 = st.columns(2)

with col6:
    genres = filtered_data['listed_in'].str.split(',').explode().str.strip()
    top_genres = genres.value_counts().head(10).sort_values()

    fig3 = px.bar(top_genres, orientation='h', 
            title="Top Genres",
            color_discrete_sequence=['#E50914'], # Netflix Red
            template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

with col7:
    top_countries = filtered_data['country'].value_counts().head(10).sort_values()

    fig4 = px.bar(top_countries, 
                  orientation='h', 
                  title="Top Countries",
                  color_discrete_sequence=['#E50914'], # Netflix Red
                  template="plotly_dark")
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
    title="Rating Distribution",
    color_discrete_sequence=['#E50914'], # Netflix Red
    template="plotly_dark"
)

st.plotly_chart(fig6, use_container_width=True)


# 1. Prepare Heatmap Data
genre_type_data = filtered_data.copy()
genre_type_data['listed_in'] = genre_type_data['listed_in'].str.split(',')
genre_type_data = genre_type_data.explode('listed_in')
genre_type_data['listed_in'] = genre_type_data['listed_in'].str.strip()

# Create the pivot table for the heatmap
heatmap_df = genre_type_data.groupby(['listed_in', 'type']).size().unstack(fill_value=0)

# Get top 20 genres by total volume to keep the heatmap readable
top_20_genres = genre_type_data['listed_in'].value_counts().head(20).index
heatmap_df = heatmap_df.loc[top_20_genres]

st.subheader("🔥 Genre Distribution: Movies vs TV Shows")

fig_heatmap = px.imshow(
    heatmap_df,
    text_auto=True, # This replaces 'annot=True' from Seaborn
    aspect="auto",
    labels=dict(x="Content Type", y="Genre", color="Count"),
    # Netflix Red Color Scale: [Low (Black/Grey) -> High (Red)]
    color_continuous_scale=[[0, '#222222'], [1, '#E50914']], 
    template="plotly_dark"
)

# Update layout for cleaner labels
fig_heatmap.update_layout(
    xaxis_title="Type",
    yaxis_title="Genre",
    coloraxis_showscale=False # Hides the side color bar for a cleaner look
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Clustering (KMeans)

import joblib
import pandas as pd

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

data_model = pd.read_csv("final_clustered_data.csv")

cluster_features = data_model[
    ['rating_enc', 'duration_clean', 'content_age', 'country_enc']
].dropna()

cluster_scaled = scaler.transform(cluster_features)

from sklearn.metrics import silhouette_score

# Silhouette Score
sil_score = silhouette_score(cluster_scaled, data_model.loc[cluster_features.index, 'cluster'])

# Inertia (comes from model)
inertia = kmeans.inertia_

st.markdown("### 📊 Clustering Evaluation")

colA, colB = st.columns(2)

colA.metric("Silhouette Score", f"{sil_score:.3f}")
colB.metric("Inertia", f"{inertia:.2f}")

import plotly.express as px

fig = px.scatter(
    data_model,
    x="duration_clean",
    y="content_age",
    color="cluster",
    title="Content Clusters",
    color_discrete_sequence=px.colors.qualitative.Plotly, # Distinct colors for clusters
    template="plotly_dark"
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
        title="Cluster Characteristics",
        color_discrete_sequence=px.colors.qualitative.Plotly, # Distinct colors for clusters
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col9:
    cluster_type = data_model.groupby(["cluster", "type"]).size().reset_index(name="count")

    fig2 = px.bar(cluster_type,
        x="cluster",
        y="count",
        color="type",
        barmode="group",
        title="Cluster vs Content Type Distribution",
        color_discrete_sequence=['#E50914', "#F5F0F0"], # Netflix Red + Dark Gray
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
st.markdown("## 🎬 Netflix Content Strategy Tool")

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

 
# =========================
# 🎯 CONTENT RECOMMENDATION PLATFORM (ADDED)
# =========================
 
st.markdown("## 🎯 Content Recommendation Platform")
 
# Genre (using his dataset column)
rec_genre = st.selectbox(
    "Choose Genre",
    data['listed_in'].str.split(',').explode().str.strip().dropna().unique()
)
 
# ⭐ User-friendly Rating Mapping
rating_map = {
    "TV-Y": "Family (TV-Y)",
    "TV-Y7": "Family (TV-Y7)",
    "G": "Family (G)",
    "PG": "Teen (PG)",
    "PG-13": "Teen (PG-13)",
    "TV-PG": "Teen (TV-PG)",
    "R": "Adult (R)",
    "TV-MA": "Adult (TV-MA)",
    "NC-17": "Adult (NC-17)"
}
 
reverse_rating_map = {v: k for k, v in rating_map.items()}
 
rec_rating_display = st.selectbox(
    "Choose Rating",
    list(reverse_rating_map.keys())
)
 
rec_rating = reverse_rating_map[rec_rating_display]
 
# Country
rec_country = st.selectbox(
    "Choose Country",
    data['country'].dropna().unique()
)
 
# =========================
# RECOMMENDATION LOGIC
# =========================
 
recommendations = data[
    data['listed_in'].str.contains(rec_genre, na=False)
]
 
if rec_rating:
    recommendations = recommendations[recommendations['rating'] == rec_rating]
 
if rec_country:
    recommendations = recommendations[recommendations['country'] == rec_country]
 
# Fallback (important)
if recommendations.empty:
    recommendations = data[data['listed_in'].str.contains(rec_genre, na=False)]
 
# Sort latest first
recommendations = recommendations.sort_values(by='release_year', ascending=False)
 
# =========================
# DISPLAY
# =========================
 
st.subheader("📌 Recommended Content")
 
if not recommendations.empty:
    for _, row in recommendations.head(10).iterrows():
        with st.container():
            st.markdown(f"### 🎬 {row['title']}")
            col1, col2 = st.columns([1, 3])
 
            with col1:
                st.write(f"⭐ {row['rating']}")
                st.write(f"🌍 {row['country']}")
 
            with col2:
                st.write(f"🎭 {row['listed_in']}")
                st.write(f"📝 {row['description']}")
 
            st.markdown("---")
else:
    if not recommendations.empty:
        for _, row in recommendations.head(10).iterrows():
            st.markdown(f"""
            <div style="background-color: #222222; padding: 20px; border-radius: 10px; border-left: 8px solid #E50914; margin-bottom: 15px;">
                <h3 style="margin: 0; color: white !important;">{row['title']}</h3>
                <p style="color: #808080; font-size: 0.9em;">{row['release_year']} | {row['rating']} | {row['country']}</p>
                <p style="color: #E5E5E5;">{row['description']}</p>
                <code style="color: #E50914; background: transparent;">{row['listed_in']}</code>
            </div>
            """, unsafe_allow_html=True)






