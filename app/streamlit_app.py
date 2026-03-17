import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/processed/dashboard_data.csv")

st.title("Netflix Content Analysis Dashboard")
st.markdown("Interactive Netflix content analysis dashboard")

# Sidebar
st.sidebar.header("Filters")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(data['main_genre'].dropna().astype(str).unique())
)

country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(data['main_country'].dropna().astype(str).unique())
)

content_type = st.sidebar.selectbox(
    "Select Content Type",
    ["All"] + list(data['type'].dropna().unique())
)
year = st.sidebar.slider(
    "Release Year",
    int(data['release_year'].min()),
    int(data['release_year'].max()),
    2010
)
# Filter logic
filtered = data.copy()

if genre != "All":
    filtered = filtered[filtered['main_genre'] == genre]

if country != "All":
    filtered = filtered[filtered['main_country'] == country]

if content_type != "All":
    filtered = filtered[filtered['type'] == content_type]

filtered = filtered[filtered['release_year'] >= year]

# Show rows
st.caption(f"Filtered rows: {len(filtered)}")

# Handle empty
if filtered.empty:
    st.warning("No data available for selected filters")
    st.stop()

# KPIs
colA, colB, colC = st.columns(3)
colA.metric("Total Titles", len(filtered))
colB.metric("Movies", len(filtered[filtered['type']=="Movie"]))
colC.metric("TV Shows", len(filtered[filtered['type']=="TV Show"]))
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Genres")

    genre_counts = filtered['main_genre'].value_counts().head(10)

    fig, ax = plt.subplots()
    ax.bar(genre_counts.index, genre_counts.values)
    plt.xticks(rotation=45)

    st.pyplot(fig)

with col2:
    st.subheader("Top Countries")

    country_counts = filtered['main_country'].value_counts().head(10)

    fig, ax = plt.subplots()
    ax.barh(country_counts.index, country_counts.values)

    st.pyplot(fig)


    col3, col4 = st.columns(2)

with col3:
    st.subheader("Rating Distribution")

    rating_counts = filtered['rating'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(rating_counts.index, rating_counts.values)
    plt.xticks(rotation=45)

    st.pyplot(fig)

with col4:
    st.subheader("Content Growth Over Time")

    year_counts = filtered['year_added'].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.plot(year_counts.index, year_counts.values, marker='o')

    st.pyplot(fig)

    st.subheader("Movie Duration Distribution")

movie_data = filtered[filtered['type'] == "Movie"]

fig, ax = plt.subplots()
ax.hist(movie_data['duration_int'], bins=20)

st.pyplot(fig)