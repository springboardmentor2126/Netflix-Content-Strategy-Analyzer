import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("📊 Netflix Content Strategy Dashboard")

# Load data
df = pd.read_csv("netflix_cleaned.csv")

# ✅ FIX: Create missing columns
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# Sidebar filters
st.sidebar.header("Filters")

content_type = st.sidebar.selectbox("Select Type", df["type"].unique())

filtered_df = df[df["type"] == content_type]
st.metric("Total Titles", len(filtered_df))

# Show data
st.subheader("Filtered Data")
st.dataframe(filtered_df.head())

# Top Countries
st.subheader("Top 10 Countries")

countries = filtered_df["country"].dropna().str.split(", ")
all_countries = [country for sublist in countries for country in sublist]

country_counts = pd.Series(all_countries).value_counts().head(10)

fig1, ax1 = plt.subplots()
country_counts.plot(kind="bar", ax=ax1)
plt.xticks(rotation=45)

st.pyplot(fig1)

# Content by Year
st.subheader("Content Added Over Years")

year_counts = filtered_df["year_added"].value_counts().sort_index()

fig2, ax2 = plt.subplots()
year_counts.plot(kind="line", ax=ax2)

st.pyplot(fig2)

# Ratings Distribution
st.subheader("Ratings Distribution")

rating_counts = filtered_df["rating"].value_counts()

fig3, ax3 = plt.subplots()
rating_counts.plot(kind="bar", ax=ax3)

st.pyplot(fig3)