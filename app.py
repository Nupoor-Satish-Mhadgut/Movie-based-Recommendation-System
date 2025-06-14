import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import zipfile
from io import BytesIO
import os

# --- Constants ---
DEFAULT_THUMBNAIL = "https://via.placeholder.com/300x450?text=No+Poster"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
YOUTUBE_API_KEY = "AIzaSyD-OV1x8vGmz-swPvEeFI3MOYMFNplb1RY"  # Replace with your actual key

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists("data/ml-latest-small/movies.csv"):
            with st.spinner("Downloading MovieLens dataset (this may take a minute)..."):
                response = requests.get(MOVIELENS_URL, timeout=30)
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall("data")
        
        movies = pd.read_csv("data/ml-latest-small/movies.csv")
        movies['genres'] = movies['genres'].str.replace('|', ' ')
        return movies
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# --- YouTube Functions ---
@st.cache_data(ttl=86400)
def get_youtube_trailer(title):
    """Get YouTube trailer link and thumbnail"""
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": f"{title} official trailer",
                "part": "snippet",
                "key": YOUTUBE_API_KEY,
                "maxResults": 1,
                "type": "video",
                "videoDuration": "short"  # Better chance of getting actual trailer
            },
            timeout=10
        )
        data = response.json()
        if data.get('items'):
            video_id = data['items'][0]['id']['videoId']
            return {
                'url': f"https://youtu.be/{video_id}",
                'thumbnail': f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"  # Higher quality thumbnail
            }
    except:
        pass
    return {
        'url': None,
        'thumbnail': DEFAULT_THUMBNAIL
    }

# --- Recommendation System ---
@st.cache_data
def prepare_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

# --- UI Components ---
def movie_card(movie):
    trailer = get_youtube_trailer(movie['title'])
    
    with st.container():
        # Poster image (centered)
        st.image(
            trailer['thumbnail'], 
            use_column_width=True,
            output_format="JPEG",
            caption=movie['title']  # Title appears centered below image
        )
        
        # Genres (centered)
        st.markdown(
            f"<p style='text-align:center; font-size:14px; margin-top:-15px;'>"
            f"{movie['genres']}"
            f"</p>", 
            unsafe_allow_html=True
        )
        
        # Trailer button (centered)
        if trailer['url']:
            st.markdown(
                f"<div style='text-align:center; margin-top:10px;'>"
                f"<a href='{trailer['url']}' target='_blank' style='"
                f"color: white; background-color: #FF0000; "
                f"padding: 8px 16px; border-radius: 4px; "
                f"text-decoration: none; font-weight: bold;'>"
                f"▶ Watch Trailer"
                f"</a></div>",
                unsafe_allow_html=True
            )

# --- Main App ---
def main():
    st.set_page_config(layout="wide")
    st.title("🎬 Nupoor Mhadgut's Movie Recommendation System")
    st.markdown("Discover similar movies with instant trailer access")
    
    movies = load_data()
    if movies is None:
        st.stop()
    
    cosine_sim = prepare_model(movies)
    
    with st.sidebar:
        st.header("Filters")
        num_recs = st.slider("Number of recommendations", 3, 10, 5)
        st.markdown("---")
        st.markdown(f"Dataset contains {len(movies)} movies")
    
    # Main content area
    selected = st.selectbox(
        "Select a movie you like:",
        movies['title'].sort_values(),
        index=movies['title'].tolist().index("Interstellar (2014)") if "Interstellar (2014)" in movies['title'].values else 0
    )
    
    if st.button("Find Similar Movies", type="primary"):
        with st.spinner("Finding recommendations..."):
            idx = movies[movies['title'] == selected].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
            
            st.subheader(f"Movies similar to: {selected}")
            
            # Create responsive columns
            cols = st.columns(min(3, num_recs))  # Max 3 columns
            
            for i, (idx, score) in enumerate(sim_scores):
                with cols[i % len(cols)]:
                    movie_card(movies.iloc[idx])

if __name__ == "__main__":
    main()