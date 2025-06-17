import streamlit as st
import streamlit.components.v1 as components
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

    card_html = f"""
    <div style="
        display: inline-block;
        width: 220px;
        margin-right: 25px;
        margin-bottom: 30px;
        vertical-align: top;
        transition: all 0.3s ease;
    ">
        <div style="
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        ">
            <img src="{poster_url}" 
                 style="
                    width: 100%;
                    height: 330px;
                    object-fit: cover;
                    transition: transform 0.5s ease;
                    display: block;
                 "
                 onerror="this.src='{DEFAULT_THUMBNAIL}'"
            >
            <div style="
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 100%);
                padding: 60px 15px 15px;
                opacity: 0;
                transition: opacity 0.3s ease;
                border-radius: 0 0 8px 8px;
            ">
                <div style="
                    font-weight: 600;
                    font-size: 16px;
                    margin-bottom: 8px;
                    color: white;
                    text-shadow: 0 1px 3px rgba(0,0,0,0.5);
                ">
                    {movie['display_title']}
                </div>
                <div style="
                    font-size: 13px;
                    color: #e0e0e0;
                    margin-bottom: 12px;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
                ">
                    {movie['year']} • {', '.join(movie['genres'].split()[:2])}
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
                        transition: all 0.2s ease;
                        border: none;
                        outline: none;
                        box-shadow: none;
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
    st.set_page_config(layout="wide", page_title="🎬 Nupoor Mhadgut's Movie Recommendation Engine", page_icon="🎥")
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .header {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            padding: 2rem 3rem;
            margin-bottom: 2rem;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .movie-grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 25px;
            padding: 20px 0;
        }
        
        .movie-card:hover {
            transform: translateY(-10px) scale(1.02);
        }
        
        .movie-card:hover .movie-poster {
            transform: scale(1.05);
        }
        
        .movie-card:hover .movie-overlay {
            opacity: 1;
        }
        
        .trailer-btn:hover {
            background: #f40612 !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(228,9,20,0.4);
        }
        
        body {
            background: linear-gradient(to bottom, #1a1a2e 0%, #16213e 100%);
            color: white;
            margin: 0;
            padding: 0;
        }
        
        .section-title {
            color: white;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 20px 0 30px 0;
            position: relative;
            display: inline-block;
        }
        
        .section-title:after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 0;
            width: 60px;
            height: 4px;
            background: #e50914;
            border-radius: 2px;
        }
        
        .select-container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 3rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .stButton>button {
            background: linear-gradient(to right, #e50914 0%, #b00710 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(229,9,20,0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(229,9,20,0.4);
        }
        
        /* Fix for select box dropdown */
        .stSelectbox div[role="listbox"] {
            max-height: 300px;
            overflow-y: auto;
            position: absolute;
            z-index: 100;
            background: white;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            width: 100%;
            top: 100%;
            margin-top: -1px;
        }
        
        /* Ensure dropdown appears below the input */
        .stSelectbox > div:first-child {
            position: relative;
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
    
    # Movie Selection Section
    with st.container():
        st.markdown('<div class="select-container">', unsafe_allow_html=True)
        
        # Create a session state to track if it's the first search
        if 'first_search' not in st.session_state:
            st.session_state.first_search = True
            
        selected = st.selectbox(
            "🎞️ SELECT A MOVIE YOU LIKE:",
            movies['title'].sort_values(),
            index=movies['title'].tolist().index("Toy Story (1995)") if "Toy Story (1995)" in movies['title'].values else 0,
            key="movie_select"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
        ">
            <h3 style="color:white; margin-top:0;">⚙️ CONTROLS</h3>
            <p style="color:rgba(255,255,255,0.7); font-size:0.9rem;">
                Customize your recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
        num_recs = st.slider("Number of recommendations", 3, 20, 6)
    
    # Use a form to handle the button click properly
    with st.form("movie_form"):
        submitted = st.form_submit_button("🔍 FIND SIMILAR MOVIES")
        
        if submitted:
            st.session_state.first_search = False
            with st.spinner("Analyzing preferences..."):
                idx = movies[movies['title'] == selected].index[0]
                sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:num_recs+1]
                
                st.markdown(f'<div class="section-title">Because you watched: <span style="color:#e50914">{movies.iloc[idx]["display_title"]}</span></div>', unsafe_allow_html=True)
                
                # Create grid layout instead of horizontal scrolling
                grid_html = '<div class="movie-grid-container">'
                for idx, score in sim_scores:
                    grid_html += movie_card(movies.iloc[idx])
                grid_html += '</div>'
                
                st.components.v1.html(grid_html, height=600)
                
                # Add footer
                st.markdown("""
                <div style="
                    text-align:center;
                    margin-top:4rem;
                    padding:2rem 0;
                    color:rgba(255,255,255,0.6);
                    font-size:0.9rem;
                    border-top:1px solid rgba(255,255,255,0.1);
                ">
                    <p>AI-Powered Movie Recommendation System</p>
                    <p>Built with Python, Streamlit, and TMDB API</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()