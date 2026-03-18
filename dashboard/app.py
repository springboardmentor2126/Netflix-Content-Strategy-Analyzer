import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="StreamScope Dashboard", layout="wide")

# -------------------------
# CUSTOM CSS (UI IMPROVEMENT)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.card h1 {
    color: #00C853;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 StreamScope: Netflix Content Strategy Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/processed/netflix_cleaned_featured.csv")

# -------------------------
# LOAD MODEL
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
le_genre = pickle.load(open("le_genre.pkl", "rb"))
le_country = pickle.load(open("le_country.pkl", "rb"))
le_rating = pickle.load(open("le_rating.pkl", "rb"))
le_type = pickle.load(open("le_type.pkl", "rb"))

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🔍 Filters")

year_range = st.sidebar.slider(
    "Release Year",
    int(df['release_year'].min()),
    int(df['release_year'].max()),
    (2010, 2020)
)

df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]

type_filter = st.sidebar.multiselect(
    "Content Type",
    df['type'].unique(),
    default=df['type'].unique()
)
df = df[df['type'].isin(type_filter)]

rating_filter = st.sidebar.multiselect(
    "Rating",
    df['rating'].unique(),
    default=df['rating'].unique()
)
df = df[df['rating'].isin(rating_filter)]

search = st.sidebar.text_input("🔎 Search Title")
if search:
    df = df[df['title'].str.contains(search, case=False, na=False)]

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "🚀 Applications"])

# =========================
# OVERVIEW TAB
# =========================
with tab1:

    st.markdown("## 📊 Dashboard Overview")

    total = len(df)
    movies = df[df['type'] == 'Movie'].shape[0]
    shows = df[df['type'] == 'TV Show'].shape[0]

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="card"><h3>Total Titles</h3><h1>{total}</h1></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card"><h3>Movies</h3><h1>{movies}</h1></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card"><h3>TV Shows</h3><h1>{shows}</h1></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick Insights
    st.markdown("### ⚡ Quick Insights")

    col4, col5 = st.columns(2)

    with col4:
        st.info(f"🎭 Top Genre: **{df['listed_in'].value_counts().idxmax()}**")

    with col5:
        st.info(f"🌍 Top Country: **{df['primary_country'].value_counts().idxmax()}**")

    st.markdown("---")

    st.markdown("### 📂 Dataset Preview")
    st.dataframe(df.head())

    st.download_button(
        "📥 Download Filtered Data",
        df.to_csv(index=False),
        file_name="filtered_netflix.csv"
    )

# =========================
# ANALYSIS TAB
# =========================
with tab2:

    st.subheader("📈 Content Growth")
    growth = df['release_year'].value_counts().sort_index()
    fig1 = px.line(x=growth.index, y=growth.values)
    fig1.update_traces(mode="lines+markers")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🎥 Content Type")
    fig2 = px.pie(df, names='type')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⭐ Ratings")
    fig3 = px.histogram(df, x="rating")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🌍 Top Countries")
    country_counts = df['primary_country'].value_counts().head(10)
    st.bar_chart(country_counts)

    st.subheader("🎭 Genres")
    genre_counts = df['listed_in'].value_counts().head(10)
    st.bar_chart(genre_counts)

    # NEW GRAPHS
    if 'date_added' in df.columns:
        st.subheader("📅 Content Added Over Time")
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        yearly_add = df['date_added'].dt.year.value_counts().sort_index()
        fig = px.line(x=yearly_add.index, y=yearly_add.values)
        st.plotly_chart(fig, use_container_width=True)

    if 'director' in df.columns:
        st.subheader("🎬 Top Directors")
        directors = df['director'].value_counts().head(10)
        fig = px.bar(x=directors.index, y=directors.values)
        st.plotly_chart(fig, use_container_width=True)

    if 'duration' in df.columns:
        st.subheader("⏱ Duration Distribution")
        fig = px.histogram(df, x="duration")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌍 Country vs Type")
    country_type = df.groupby(['primary_country', 'type']).size().reset_index(name='count')
    top_countries = df['primary_country'].value_counts().head(5).index
    country_type = country_type[country_type['primary_country'].isin(top_countries)]

    fig = px.bar(
        country_type,
        x="primary_country",
        y="count",
        color="type",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Genre vs Rating Heatmap")
    heatmap_data = pd.crosstab(df['listed_in'], df['rating'])
    fig = px.imshow(heatmap_data)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# APPLICATIONS TAB
# =========================
with tab3:

    st.header("📊 Content Investment Planning")

    st.subheader("🎭 Top Genres")
    genre_counts = df['listed_in'].value_counts().head(10)
    st.bar_chart(genre_counts)
    st.success(f"👉 Invest more in **{genre_counts.idxmax()}**")

    st.subheader("🌍 Top Countries")
    country_counts = df['primary_country'].value_counts().head(10)
    st.bar_chart(country_counts)
    st.success(f"👉 Expand in **{country_counts.idxmax()}**")

    st.subheader("🎬 Content Strategy")
    type_counts = df['type'].value_counts()
    st.bar_chart(type_counts)

    # -------------------------
    # ML PREDICTION
    # -------------------------
    st.markdown("---")
    st.header("🤖 Predict Content Type")

    genre_input = st.selectbox("Genre", le_genre.classes_)
    country_input = st.selectbox("Country", le_country.classes_)
    rating_input = st.selectbox("Rating", le_rating.classes_)

    if st.button("Predict"):
        g = le_genre.transform([genre_input])[0]
        c = le_country.transform([country_input])[0]
        r = le_rating.transform([rating_input])[0]

        pred = model.predict([[g, c, r]])
        result = le_type.inverse_transform(pred)[0]

        st.success(f"🎯 Predicted Content Type: {result}")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 StreamScope Project | Data Visualization")