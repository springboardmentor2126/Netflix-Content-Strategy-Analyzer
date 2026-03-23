import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="StreamScope: Netflix Strategy Analyzer", 
    layout="wide", 
    page_icon="🎬"
)

# --- 2. Robust Data Loading ---
@st.cache_data
def load_data():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    data_path = os.path.join(project_root, "data", "netflix_cleaned_final.csv")
    
    if not os.path.exists(data_path):
        st.error(f"Error: File not found at {data_path}")
        st.stop()
        
    df = pd.read_csv(data_path)
    # Data Cleaning for consistency
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(2000).astype(int)
    if 'duration_num' not in df.columns:
        df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float).fillna(0)
    
    df['country'] = df['country'].fillna("Unknown")
    return df

df = load_data()

# --- 3. Sidebar Navigation & Insights ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.header("Navigation & Filters")

# Executive Summary for Presentation
with st.sidebar.expander("📊 Key Project Insights"):
    st.write("- **Themes:** content highlights 'Life', 'Family', and 'International' drama.")
    st.write("- **Pivot:** Significant library growth observed post-2015.")
    st.write("- **Market:** USA and India lead as the primary production hubs.")

# Global Filters
search_query = st.sidebar.text_input("🔍 Search Title", "")
all_countries = sorted(df['country'].str.split(', ').explode().unique())
selected_countries = st.sidebar.multiselect("Filter by Country", all_countries)
year_range = st.sidebar.slider("Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2015, 2024))

# Filtered Data Logic
filtered_df = df[df['release_year'].between(year_range[0], year_range[1])]
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].str.contains('|'.join(selected_countries), na=False)]
if search_query:
    filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]

# --- 4. Main UI Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Visual Storytelling", "🔍 Smart Recommendation", "🎬 Perfect Night Planner"])

# --- TAB 1: VISUAL STORYTELLING ---
with tab1:
    st.title("Strategic Content Storytelling")
    
    # KPI Section
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Library Items", len(filtered_df))
    k2.metric("Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
    k3.metric("TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))
    
    st.divider()
    
    # Row 1: Word Cloud & Text Insight
    col_wc, col_insight = st.columns([2, 1])
    with col_wc:
        st.subheader("Common Themes in Descriptions")
        text = " ".join(df['description'].dropna())
        wc = WordCloud(background_color='white', colormap='Reds', width=800, height=400).generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    
    with col_insight:
        st.markdown("### 📝 Text Analysis")
        st.info("""
        The Word Cloud highlights that Netflix prioritizes **Family, Life, and Mystery**-centric plots. 
        This reveals a strategy of creating 'broad-appeal' content for a global audience.
        """)

    st.divider()
    
    # Row 2: NEW ENHANCEMENT - Growth Chart
    st.subheader("📈 Content Growth Over Time")
    growth = filtered_df.groupby('release_year').size().reset_index(name='count')
    fig_growth = px.area(growth, x='release_year', y='count', color_discrete_sequence=['#E50914'])
    fig_growth.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_growth, use_container_width=True)
    st.info("💡 **Observation:** The rapid spike post-2015 highlights Netflix's pivot toward massive original content production.")

    st.divider()
    
    # Row 3: Distribution Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Content Type Split")
        fig_pie = px.pie(filtered_df, names='type', hole=0.5, color_discrete_sequence=['#E50914', '#221F1F'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        st.subheader("Top 10 Genres")
        top_genres = filtered_df['listed_in'].str.split(', ').explode().value_counts().head(10)
        fig_bar = px.bar(top_genres, x=top_genres.values, y=top_genres.index, orientation='h', color_discrete_sequence=['#E50914'])
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: SMART RECOMMENDATION ---
with tab2:
    st.header("Tailored Recommendations")
    all_genres = sorted(df['listed_in'].str.split(', ').explode().unique())
    
    r1, r2 = st.columns(2)
    with r1:
        rec_genre = st.selectbox("Pick a Genre", all_genres)
    with r2:
        rec_type = st.radio("Format", ["Movie", "TV Show"], horizontal=True)
        
    recommendations = df[(df['listed_in'].str.contains(rec_genre, na=False)) & (df['type'] == rec_type)].head(5)
    st.dataframe(recommendations[['title', 'release_year', 'duration', 'description']], hide_index=True)

# --- TAB 3: PERFECT NIGHT PLANNER ---
with tab3:
    st.header("🎬 The Perfect Night Planner")
    
    p1, p2 = st.columns(2)
    with p1:
        watch_time = st.slider("How much time do you have? (minutes)", 20, 240, 90)
    with p2:
        mood = st.selectbox("Who are you with?", ["Just Me", "Kids", "Friends", "Date Night"])
    
    # NEW ENHANCEMENT: Smart Context Message
    st.markdown("---")
    if watch_time < 60:
        st.subheader("You have time for a quick episode or short content 📺")
        plan_df = df[df['type'] == 'TV Show']
    else:
        st.subheader("You have enough time for a full-length movie 🎬")
        plan_df = df[df['type'] == 'Movie']
    
    # Mood Mapping Logic
    mood_map = {
        "Kids": "Children|Family",
        "Date Night": "Romance|Drama",
        "Friends": "Comedy|Action|Horror",
        "Just Me": "" # Empty means no filter (Fixed "Just Me" Issue)
    }
    
    query = mood_map[mood]
    if query:
        final_selection = plan_df[plan_df['listed_in'].str.contains(query, case=False, na=False)]
    else:
        final_selection = plan_df

    # Display Results
    if not final_selection.empty:
        # Show 3 random results
        results = final_selection.sample(min(3, len(final_selection)))
        for _, row in results.iterrows():
            with st.expander(f"🍿 {row['title']} ({row['release_year']})"):
                st.write(f"**Duration:** {row['duration']} | **Genres:** {row['listed_in']}")
                st.write(f"**Plot:** {row['description']}")
    else:
        st.warning("No exact matches found. How about these popular picks?")
        st.table(plan_df.sample(3)[['title', 'release_year', 'duration']])

# --- 5. Export Feature ---
st.sidebar.divider()
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Export Current View", data=csv, file_name='streamscope_data.csv')