import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import zipfile
from io import BytesIO
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import html

# --- Constants ---
DEFAULT_THUMBNAIL = "https://via.placeholder.com/300x450.png?text=No+Poster"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# --- API Configuration ---
if 'api_keys' not in st.secrets:
    st.error("❌ API keys missing in Streamlit secrets!")
    st.stop()

YOUTUBE_API_KEY = st.secrets["api_keys"].get("YOUTUBE_API_KEY")
OMDB_API_KEY = st.secrets["api_keys"].get("OMDB_API_KEY")

# --- Recommendation System ---
def prepare_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists("data/ml-latest-small/movies.csv"):
            with st.spinner("📦 Downloading MovieLens dataset..."):
                response = requests.get(MOVIELENS_URL, timeout=30)
                response.raise_for_status()
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall("data")
        
        movies = pd.read_csv("data/ml-latest-small/movies.csv")
        movies['genres'] = movies['genres'].str.replace('|', ' ')
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['display_title'] = movies['title'].apply(
            lambda x: html.escape(re.sub(r'\(\d{4}\)', '', x).strip())
        )
        return movies
    except Exception as e:
        st.error(f"🚨 Error loading data: {str(e)}")
        return None

# --- API Functions ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_youtube_trailer(title):
    if not YOUTUBE_API_KEY:
        return None
    try:
        clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": f"{clean_title} official trailer",
                "part": "snippet",
                "key": YOUTUBE_API_KEY,
                "maxResults": 1,
                "type": "video",
                "videoDuration": "short",
                "videoEmbeddable": "true"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return f"https://youtu.be/{data['items'][0]['id']['videoId']}" if data.get('items') else None
    except Exception:
        return None

def fetch_itunes_trailer(title, year=None):
    try:
        clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
        query = f"{clean_title} {year}" if year else clean_title
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "media": "movie",
                "entity": "movie",
                "limit": 1,
                "country": "US"
            },
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data["results"][0].get("previewUrl") if data.get("resultCount", 0) > 0 else None
    except Exception:
        return None

def get_best_trailer(title, year):
    youtube_url = fetch_youtube_trailer(title)
    if youtube_url:
        return {
            'url': youtube_url,
            'source': 'youtube',
            'button_color': '#FF0000',
            'badge': 'YouTube'
        }
    
    itunes_url = fetch_itunes_trailer(title, year)
    if itunes_url:
        return {
            'url': itunes_url,
            'source': 'itunes',
            'button_color': '#000000',
            'badge': 'iTunes'
        }
    return None

def fetch_poster(title, year):
    if OMDB_API_KEY:
        try:
            clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
            response = requests.get(
                "http://www.omdbapi.com/",
                params={
                    "t": clean_title,
                    "y": year,
                    "apikey": OMDB_API_KEY,
                    "r": "json"
                },
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            if data.get('Poster') and data['Poster'] != 'N/A':
                return data['Poster']
        except Exception:
            pass
    
    try:
        clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
        query = f"{clean_title} {year}" if year else clean_title
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "media": "movie",
                "entity": "movie",
                "limit": 1,
                "country": "US"
            },
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if data.get("resultCount", 0) > 0:
            return data["results"][0].get("artworkUrl100", "").replace("100x100bb", "600x600bb")
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_movie_media(movie):
    poster = fetch_poster(movie['title'], movie.get('year'))
    return {
        'poster': poster if poster else DEFAULT_THUMBNAIL,
        'trailer': get_best_trailer(movie['title'], movie.get('year'))
    }

def movie_card(movie):
    media = get_movie_media(movie)
    
    card_html = f"""
    <div style='
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
        background: white;
        display: flex;
        flex-direction: column;
    '>
        <img src='{media["poster"]}' 
             style='
                 width: 100%;
                 height: auto;
                 border-radius: 8px;
                 aspect-ratio: 2/3;
                 object-fit: contain;
                 background: #f5f5f5;
             '>
        
        <h3 style='
            text-align: center;
            margin: 12px 0 4px;
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
        '>
            {movie['display_title']} ({movie['year']})
        </h3>
        
        <div style='
            text-align: center; 
            margin: 4px 0 12px; 
            color: #666; 
            font-size: 0.9rem;
            min-height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
        '>
            {', '.join(movie['genres'].split()[:3])}
        </div>
    """
    
    if media['trailer']:
        card_html += f"""
        <div style='text-align: center; margin-top: auto;'>
            <a href='{media["trailer"]["url"]}' target='_blank' 
            style='
                display: inline-block;
                background: {media["trailer"]["button_color"]};
                color: white;
                padding: 8px 16px;
                border-radius: 50px;
                text-decoration: none;
                font-weight: 600;
                font-size: 0.9rem;
            '>
                ▶ Watch Trailer ({media["trailer"]["badge"]})
            </a>
        </div>
        """
    
    card_html += "</div>"
    st.components.v1.html(card_html, height=450)

def main():
    st.set_page_config(
        layout="wide",
        page_title="🎬 Movie Recommendation Engine",
        page_icon="🎥"
    )
    
    # Custom CSS with stronger dropdown arrow styling
    st.markdown("""
    <style>
        /* Force permanent downward arrow */
        div[data-baseweb="select"] > div:first-child > div:after {
            content: "▼" !important;
            position: absolute !important;
            top: 50% !important;
            right: 0.5rem !important;
            transform: translateY(-50%) !important;
            pointer-events: none !important;
            font-size: 12px !important;
        }
        
        /* Prevent arrow flip when open */
        div[data-baseweb="select"] > div:first-child > div[aria-expanded="true"]:after {
            transform: translateY(-50%) rotate(0deg) !important;
        }
        
        /* Header styling */
        .header {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        }
    </style>
    <div class="header">
        <h1 style='text-align: center; margin: 0;'>🎬 Movie Recommendation Engine</h1>
        <p style='text-align: center; margin: 0.5rem 0 0;'>Discover your next favorite film</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    movies = load_data()
    if movies is None:
        st.stop()
    
    cosine_sim = prepare_model(movies)
    
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        num_recs = st.slider("Number of recommendations", 3, 10, 5)
        st.markdown("---")
        st.markdown("Built with ❤️ by Nupoor Mhadgut")
    
    # Movie selection with permanent downward arrow
    selected = st.selectbox(
        "🎞️ Select a movie you like:",
        movies['title'].sort_values(),
        index=movies['title'].tolist().index("Toy Story (1995)") if "Toy Story (1995)" in movies['title'].values else 0,
        help="Search from 9,000+ movies"
    )
    
    if st.button("🔍 Find Similar Movies", type="primary"):
        with st.spinner("Finding recommendations..."):
            idx = movies[movies['title'] == selected].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
            
            st.markdown(f"## 🎯 Similar to: **{movies.iloc[idx]['display_title']} ({movies.iloc[idx]['year']})**")
            cols = st.columns(min(3, len(sim_scores)))
            
            for i, (idx, score) in enumerate(sim_scores):
                with cols[i % len(cols)]:
                    movie_card(movies.iloc[idx])

if __name__ == "__main__":
    main()