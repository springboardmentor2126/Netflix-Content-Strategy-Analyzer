import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

st.title("🎬 Netflix Data Analytics Dashboard")

# ----------------------
# LOAD DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/netflix_cleaned.csv")
    df.drop_duplicates(inplace=True)
    df.fillna("Unknown", inplace=True)
    df.columns = df.columns.str.lower()
    df["type"] = df["type"].str.strip().str.title()
    return df

df = load_data()

# ----------------------
# SIDEBAR FILTERS
# ----------------------
st.sidebar.header("🔍 Filters")

type_filter = st.sidebar.multiselect(
    "Select Type",
    options=df["type"].unique(),
    default=df["type"].unique()
)

country_filter = st.sidebar.multiselect(
    "Select Country",
    options=df["country"].unique(),
    default=df["country"].unique()
)

year_filter = st.sidebar.slider(
    "Select Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (2000, 2021)
)
# ✅ ADD THIS HERE
genre_filter = st.sidebar.multiselect(
    "Select Genre",
    options=df["listed_in"].str.split(", ").explode().unique()
)

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["country"].isin(country_filter)) &
    (df["release_year"].between(year_filter[0], year_filter[1])) &
    (df["listed_in"].str.contains('|'.join(genre_filter)) if genre_filter else True)
]

# KPIs
# ----------------------
st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies", len(filtered_df[filtered_df["type"].str.contains("Movie", case=False)]))
col3.metric("TV Shows", len(filtered_df[filtered_df["type"].str.contains("TV", case=False)]))
col4.metric("Top Country", filtered_df["country"].mode()[0])

# ----------------------
# GROWTH OVER TIME
# ----------------------
st.subheader("📈 Content Growth Over Time")

year_data = filtered_df.groupby("release_year").size().reset_index(name="count")

fig = px.line(year_data, x="release_year", y="count", title="Content Growth")
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# GENRE ANALYSIS
# ----------------------
st.subheader("🎭 Genre Distribution")
# ----------------------
# GENRE TREND OVER TIME
# ----------------------
st.subheader("📅 Top Genres Over Time")

genre_year = df.copy()
genre_year["genre"] = genre_year["listed_in"].str.split(", ")
genre_year = genre_year.explode("genre")

top_genre_year = genre_year.groupby(["release_year", "genre"]).size().reset_index(name="count")

fig = px.line(
    top_genre_year,
    x="release_year",
    y="count",
    color="genre",
    title="Genre Trends Over Time"
)

st.plotly_chart(fig, use_container_width=True)
genre_data = filtered_df["listed_in"].str.split(", ").explode()
genre_count = genre_data.value_counts().head(10)

fig2 = px.bar(
    x=genre_count.values,
    y=genre_count.index,
    orientation="h",
    title="Top Genres"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------
# CONTENT TYPE PIE
# ----------------------
st.subheader("🎬 Content Type Split")
# ----------------------
# RATING ANALYSIS
# ----------------------
st.subheader("⭐ Rating Distribution")

rating_count = filtered_df["rating"].value_counts()

fig = px.bar(
    x=rating_count.index,
    y=rating_count.values,
    title="Ratings Distribution"
)

st.plotly_chart(fig, use_container_width=True)
type_count = filtered_df["type"].value_counts()

fig3 = px.pie(
    values=type_count.values,
    names=type_count.index,
    title="Movies vs TV Shows"
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------------
# COUNTRY MAP 🌍
# ----------------------
st.subheader("🌍 Content by Country")

country_data = filtered_df["country"].value_counts().reset_index()
country_data.columns = ["country", "count"]

fig4 = px.choropleth(
    country_data,
    locations="country",
    locationmode="country names",
    color="count",
    title="Content Distribution by Country"
)

st.plotly_chart(fig4, use_container_width=True)

# ----------------------
# MACHINE LEARNING
# ----------------------
st.subheader("🤖 ML: Movie vs TV Classification")

ml_df = df.copy()

# Encoding
le = LabelEncoder()
ml_df["type"] = le.fit_transform(ml_df["type"])
ml_df["rating"] = le.fit_transform(ml_df["rating"])
ml_df["country"] = le.fit_transform(ml_df["country"])

X = ml_df[["release_year", "rating", "country"]]
y = ml_df["type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.write(f"✅ Model Accuracy: {round(accuracy*100, 2)}%")

# ----------------------
# FEATURE IMPORTANCE
# ----------------------
st.subheader("📌 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig5 = px.bar(
    x=features,
    y=importance,
    title="Feature Importance"
)

st.plotly_chart(fig5, use_container_width=True)

# ----------------------
# INSIGHTS SECTION
# ----------------------
st.subheader("💡 Key Insights")

st.write("""
- 📈 Content increased rapidly after 2015.
- 🎭 Drama and International genres dominate Netflix.
- 🌍 USA contributes the most content.
- 🎬 Movies are more than TV Shows.
- 🤖 ML model helps classify content effectively.
""")