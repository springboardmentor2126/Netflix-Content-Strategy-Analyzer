import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

st.title("🎬 Netflix Content Analysis & Recommendation System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/Cleaned_netflix_titles_with_clusters.csv")

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔍 Filters")

countries = st.sidebar.multiselect(
    "Select Country",
    df['country'].dropna().unique()
)

genres = st.sidebar.multiselect(
    "Select Genre",
    df['primary_genre'].dropna().unique()
)

ratings = st.sidebar.multiselect(
    "Select Rating",
    df['rating'].dropna().unique()
)

years = st.sidebar.slider(
    "Select Release Year",
    int(df['release_year'].min()),
    int(df['release_year'].max()),
    (2000, 2020)
)

# =========================
# FILTER DATA
# =========================
filtered_df = df.copy()

if countries:
    filtered_df = filtered_df[filtered_df['country'].isin(countries)]

if genres:
    filtered_df = filtered_df[filtered_df['primary_genre'].isin(genres)]

if ratings:
    filtered_df = filtered_df[filtered_df['rating'].isin(ratings)]

filtered_df = filtered_df[
    (filtered_df['release_year'] >= years[0]) &
    (filtered_df['release_year'] <= years[1])
]

# =========================
# KPI METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", filtered_df.shape[0])
col2.metric("Movies", filtered_df[filtered_df['type'] == 'Movie'].shape[0])
col3.metric("TV Shows", filtered_df[filtered_df['type'] == 'TV Show'].shape[0])

top_genre = filtered_df['primary_genre'].mode()
col4.metric("Top Genre", top_genre[0] if not top_genre.empty else "N/A")

# =========================
# CHARTS (2x2 GRID)
# =========================

type_counts = filtered_df['type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']
fig1 = px.pie(type_counts, names='Type', values='Count', title="Content Distribution")

growth = filtered_df.groupby('year_added').size().reset_index(name='count')
fig2 = px.line(growth, x='year_added', y='count', title="Content Growth Over Time")

genre_counts = filtered_df['primary_genre'].value_counts().head(10).reset_index()
genre_counts.columns = ['Genre', 'Count']
fig3 = px.bar(genre_counts, x='Genre', y='Count', title="Top Genres")

rating_counts = filtered_df['rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Count']
fig4 = px.bar(rating_counts, x='Rating', y='Count', title="Rating Distribution")

colA, colB = st.columns(2)
with colA:
    st.plotly_chart(fig1, use_container_width=True)
with colB:
    st.plotly_chart(fig2, use_container_width=True)

colC, colD = st.columns(2)
with colC:
    st.plotly_chart(fig3, use_container_width=True)
with colD:
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# CLUSTER VISUALIZATION
# =========================
st.subheader("🧠 Content Clusters")

fig_cluster = px.scatter(
    filtered_df,
    x='release_year',
    y='duration_num',
    color='cluster',
    hover_data=['title', 'primary_genre'],
    title="Content Clustering"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# =========================
# SEARCH FUNCTION
# =========================
st.subheader("🔎 Search Content")

search_query = st.text_input("Search by title")

if search_query:
    results = df[df['title'].str.contains(search_query, case=False, na=False)]

    st.subheader("🎬 Search Results")

    if not results.empty:
        for _, row in results.head(10).iterrows():
            with st.container():
                st.markdown(f"### 🎬 {row['title']}")
                st.write(f"🎭 Genre: {row['primary_genre']}")
                st.write(f"⭐ Rating: {row['rating']}")
                st.write(f"🌍 Country: {row['country']}")
                st.write(f"📝 {row['description']}")
                st.markdown("---")
    else:
        st.write("No matching titles found.")

# =========================
# RECOMMENDATION SYSTEM
# =========================
st.subheader("🎯 Content Recommendation Platform")

rec_genre = st.selectbox("Choose Genre", df['primary_genre'].unique())

# ⭐ USER-FRIENDLY RATING DROPDOWN
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

rec_country = st.selectbox("Choose Country", df['country'].unique())

recommendations = df[
    (df['primary_genre'] == rec_genre) &
    (df['rating'] == rec_rating) &
    (df['country'] == rec_country)
]

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
                st.write(f"🎭 {row['primary_genre']}")
                st.write(f"📝 {row['description']}")

            st.markdown("---")
else:
    st.write("No recommendations found. Try different filters.")