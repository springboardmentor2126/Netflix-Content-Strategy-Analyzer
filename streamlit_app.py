import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dashboard_data.csv")

st.title("Netflix Content Analysis Dashboard")

# =========================
# Sidebar Filters
# =========================

st.sidebar.header("Filters")

genre = st.sidebar.selectbox(
    "Select Genre",
    sorted(data['main_genre'].dropna().astype(str).unique())
)

country = st.sidebar.selectbox(
    "Select Country",
    sorted(data['main_country'].dropna().astype(str).unique())
)

content_type = st.sidebar.selectbox(
    "Select Content Type",
    data['type'].dropna().unique()
)

year = st.sidebar.slider(
    "Release Year",
    int(data['release_year'].min()),
    int(data['release_year'].max()),
    int(data['release_year'].min())
)

# =========================
# Apply Filters
# =========================

filtered = data[
    (data['main_genre'] == genre) &
    (data['main_country'] == country) &
    (data['type'] == content_type) &
    (data['release_year'] >= year)
]

st.write("Filtered Data Shape:", filtered.shape)

# =========================
# First Row Charts
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Genres")

    genre_counts = filtered['main_genre'].value_counts().head(10)

    fig, ax = plt.subplots()

    ax.bar(genre_counts.index, genre_counts.values)

    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45)

    st.pyplot(fig)

with col2:
    st.subheader("Top Countries")

    country_counts = filtered['main_country'].value_counts().head(10)

    fig, ax = plt.subplots()

    ax.barh(country_counts.index, country_counts.values)

    ax.set_xlabel("Count")
    ax.set_ylabel("Country")

    st.pyplot(fig)

# =========================
# Second Row Charts
# =========================

col3, col4 = st.columns(2)

with col3:
    st.subheader("Rating Distribution")

    rating_counts = filtered['rating'].value_counts()

    fig, ax = plt.subplots()

    ax.bar(rating_counts.index, rating_counts.values)

    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45)

    st.pyplot(fig)

with col4:
    st.subheader("Content Growth Over Time")

    year_counts = filtered['year_added'].value_counts().sort_index()

    fig, ax = plt.subplots()

    ax.plot(year_counts.index, year_counts.values, marker='o')

    ax.set_xlabel("Year")
    ax.set_ylabel("Titles")

    st.pyplot(fig)

# =========================
# Duration Distribution
# =========================

st.subheader("Movie Duration Distribution")

movie_data = filtered[filtered['type'] == "Movie"]

fig, ax = plt.subplots()

ax.hist(movie_data['duration_int'], bins=20)

ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Frequency")

st.pyplot(fig)