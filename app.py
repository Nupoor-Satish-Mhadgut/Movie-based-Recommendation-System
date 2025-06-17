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
import logging

# Configure logging
logging.basicConfig(filename='movie_errors.log', level=logging.INFO)

# Constants
DEFAULT_THUMBNAIL = "https://via.placeholder.com/300x450.png?text=No+Poster"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Check for API keys
if 'api_keys' not in st.secrets:
    st.error("❌ API keys missing in Streamlit secrets!")
    st.stop()

TMDB_API_KEY = st.secrets["api_keys"].get("TMDB_API_KEY")
YOUTUBE_API_KEY = st.secrets["api_keys"].get("YOUTUBE_API_KEY")
OMDB_API_KEY = st.secrets["api_keys"].get("OMDB_API_KEY")

def prepare_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

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
            lambda x: html.escape(re.sub(r'\(\d{4}\)', '', x).strip()))
        return movies
    except Exception as e:
        st.error(f"🚨 Error loading data: {str(e)}")
        return None

def fetch_poster(title, year):
    clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
    
    # 1. Try TMDB first
    if TMDB_API_KEY:
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            search_params = {
                "api_key": TMDB_API_KEY,
                "query": clean_title,
                "year": year,
                "language": "en-US"
            }
            search_response = requests.get(search_url, params=search_params, timeout=3)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if search_data.get('results') and len(search_data['results']) > 0:
                movie_id = search_data['results'][0]['id']
                movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                movie_params = {"api_key": TMDB_API_KEY}
                movie_response = requests.get(movie_url, params=movie_params, timeout=3)
                movie_response.raise_for_status()
                movie_data = movie_response.json()
                
                if movie_data.get('poster_path'):
                    return f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}"
        except Exception as e:
            logging.info(f"TMDB poster failed for {title}: {str(e)}")
    
    # 2. Fallback to OMDB
    if OMDB_API_KEY:
        try:
            response = requests.get(
                "http://www.omdbapi.com/",
                params={"t": clean_title, "y": year, "apikey": OMDB_API_KEY},
                timeout=3
            )
            data = response.json()
            if data.get('Poster') not in [None, 'N/A']:
                return data['Poster']
        except Exception as e:
            logging.info(f"OMDB failed for {title}: {str(e)}")

    # 3. Fallback to iTunes
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": f"{clean_title} {year}",
                "media": "movie",
                "limit": 1
            },
            timeout=3
        )
        data = response.json()
        if data.get("resultCount", 0) > 0:
            artwork = data["results"][0].get("artworkUrl100", "")
            if artwork:
                return artwork.replace("100x100bb", "600x600bb")
    except Exception as e:
        logging.info(f"iTunes failed for {title}: {str(e)}")

    # 4. Final fallback - placeholder with title
    title_text = f"{clean_title}+{year}" if year else clean_title
    return f"https://via.placeholder.com/300x450/6e48aa/ffffff.png?text={title_text.replace(' ', '+')}"

def fetch_tmdb_trailer(title, year=None):
    if not TMDB_API_KEY:
        return None
        
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        search_params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "year": year,
            "language": "en-US"
        }
        search_response = requests.get(search_url, params=search_params, timeout=5)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if not search_data.get('results'):
            return None
            
        movie_id = search_data['results'][0]['id']
        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        videos_params = {"api_key": TMDB_API_KEY}
        videos_response = requests.get(videos_url, params=videos_params, timeout=5)
        videos_response.raise_for_status()
        videos_data = videos_response.json()
        
        for video in videos_data.get('results', []):
            if video['site'] == 'YouTube' and video['type'] == 'Trailer':
                return {
                    'url': f"https://youtu.be/{video['key']}",
                    'source': 'youtube',
                    'button_color': '#FF0000'
                }
                
    except Exception as e:
        logging.info(f"TMDB trailer failed for {title}: {str(e)}")
    return None

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
                "type": "video"
            },
            timeout=10
        )
        data = response.json()
        return f"https://youtu.be/{data['items'][0]['id']['videoId']}" if data.get('items') else None
    except Exception as e:
        logging.info(f"YouTube failed for {title}: {str(e)}")
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
        data = response.json()
        return data["results"][0].get("previewUrl") if data.get("resultCount", 0) > 0 else None
    except Exception as e:
        logging.info(f"iTunes trailer failed for {title}: {str(e)}")
        return None

def get_best_trailer(title, year):
    tmdb_trailer = fetch_tmdb_trailer(title, year)
    if tmdb_trailer:
        return tmdb_trailer
    
    youtube_url = fetch_youtube_trailer(title)
    if youtube_url: 
        return {
            'url': youtube_url, 
            'source': 'youtube', 
            'button_color': '#FF0000'
        }
    
    itunes_url = fetch_itunes_trailer(title, year)
    if itunes_url:
        return {
            'url': itunes_url,
            'source': 'itunes',
            'button_color': '#000000'
        }
    
    return None

@st.cache_data(ttl=3600)
def get_movie_media(movie):
    return {
        'poster': fetch_poster(movie['title'], movie.get('year')),
        'trailer': get_best_trailer(movie['title'], movie.get('year'))
    }

def movie_card(movie):
    media = get_movie_media(movie)
    poster_url = media.get("poster", DEFAULT_THUMBNAIL)
    trailer = media.get("trailer")

    # Escape HTML in title to prevent XSS
    safe_title = html.escape(movie['display_title'])
    safe_genres = html.escape(', '.join(movie['genres'].split()[:2]))
    safe_year = html.escape(str(movie['year'])) if pd.notna(movie['year']) else "Unknown"

    card_html = f"""
    <div style="
        display: inline-block;
        width: 220px;
        margin-right: 25px;
        margin-bottom: 30px;
        vertical-align: top;
    ">
        <div style="
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            background: #2a2a40;
        ">
            <img src="{poster_url}" 
                 style="
                    width: 100%;
                    height: 330px;
                    object-fit: cover;
                    display: block;
                 "
                 onerror="this.src='{DEFAULT_THUMBNAIL}'"
            >
            <div style="
                padding: 15px;
                background: rgba(0,0,0,0.8);
            ">
                <div style="
                    font-weight: 600;
                    font-size: 16px;
                    margin-bottom: 8px;
                    color: white;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">
                    {safe_title}
                </div>
                <div style="
                    font-size: 13px;
                    color: #e0e0e0;
                    margin-bottom: 12px;
                ">
                    {safe_year} • {safe_genres}
                </div>
                {f'''
                <a href="{trailer['url']}" target="_blank"
                   style="
                        display: inline-block;
                        background: #e50914;
                        color: white;
                        padding: 8px 16px;
                        border-radius: 4px;
                        font-size: 13px;
                        text-decoration: none !important;
                        font-weight: 500;
                   ">
                    ▶ Play Trailer
                </a>
                ''' if trailer else ''}
            </div>
        </div>
    </div>
    """
    return card_html

def main():
    st.set_page_config(layout="wide", page_title="🎬 Movie Recommendation Engine", page_icon="🎥")
    
    st.markdown("""
    <style>
        /* Fixed dropdown behavior */
        div[data-baseweb="select"] > div {
            z-index: 999;
        }
        
        div[data-baseweb="popover"] {
            z-index: 1000;
            position: absolute !important;
            top: 100% !important;
            bottom: auto !important;
        }
        
        /* Movie row styling */
        .movie-row {
            display: flex;
            overflow-x: auto;
            padding: 20px 0;
            gap: 25px;
            scrollbar-width: thin;
        }
        
        /* Remove extra padding */
        .stApp {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton>button {
            transition: all 0.3s ease;
        }
        
        .stButton>button:active {
            transform: scale(0.98);
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="header">
        <h1 style="
            text-align:center;
            margin:0;
            color:white;
            font-size:2.5rem;
            font-weight:700;
            letter-spacing:1px;
        ">
            🎬 Nupoor Mhadgut's Movie Recommendation Engine
        </h1>
        <p style="
            text-align:center;
            color:rgba(255,255,255,0.8);
            margin:0.5rem 0 0;
            font-size:1.1rem;
        ">
            Discover your next favorite movie with AI-powered recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    movies = load_data()
    if movies is None: 
        st.stop()
    
    cosine_sim = prepare_model(movies)
    
    with st.container():
        selected = st.selectbox(
            "🎞️ SELECT A MOVIE YOU LIKE:",
            movies['title'].sort_values(),
            index=movies['title'].tolist().index("Toy Story (1995)") if "Toy Story (1995)" in movies['title'].values else 0,
            key="movie_select"
        )
    
    with st.sidebar:
        num_recs = st.slider("Number of recommendations", 3, 20, 6)
    
    if st.button("🔍 FIND SIMILAR MOVIES", key="find_movies"):
        st.session_state.button_clicked = True
    
    if st.session_state.button_clicked:
        with st.spinner("Analyzing preferences..."):
            idx = movies[movies['title'] == selected].index[0]
            sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:num_recs+1]
            
            st.markdown(f'<div class="section-title">Because you watched: <span style="color:#e50914">{html.escape(movies.iloc[idx]["display_title"])}</span></div>', unsafe_allow_html=True)
            
            # Create horizontal movie row
            cols = st.columns(1)
            with cols[0]:
                row_html = '<div class="movie-row">'
                for idx, score in sim_scores:
                    row_html += movie_card(movies.iloc[idx])
                row_html += '</div>'
                st.markdown(row_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()