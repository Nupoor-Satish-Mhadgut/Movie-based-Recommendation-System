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

    components.html(f"""
    <style>
        .movie-card {{
            width: 160px;  /* Slightly wider than Netflix */
            margin: 0 15px 25px 0;
            position: relative;
            transition: all 0.3s ease;
            display: inline-block;
        }}
        .movie-card:hover {{
            transform: scale(1.1);
            z-index: 10;
        }}
        .movie-poster {{
            width: 100%;
            height: 240px;  /* Standard streaming service aspect ratio */
            object-fit: cover;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .movie-info {{
            opacity: 0;
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 100%);
            color: white;
            padding: 15px 10px 10px;
            border-radius: 0 0 4px 4px;
            transition: opacity 0.3s ease;
        }}
        .movie-card:hover .movie-info {{
            opacity: 1;
        }}
        .movie-title {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .movie-meta {{
            font-size: 11px;
            color: #d2d2d2;
            margin-bottom: 8px;
        }}
        .trailer-btn {{
            background: #e50914;
            color: white;
            border: none;
            padding: 6px 0;
            border-radius: 3px;
            font-size: 12px;
            width: 100%;
            text-align: center;
            display: block;
            text-decoration: none;
        }}
    </style>

    <div class="movie-card">
        <img src="{poster_url}" class="movie-poster" onerror="this.src='{DEFAULT_THUMBNAIL}'">
        <div class="movie-info">
            <div class="movie-title">{movie['display_title']}</div>
            <div class="movie-meta">{movie['year']} • {', '.join(movie['genres'].split()[:2])}</div>
            {f'<a href="{trailer["url"]}" target="_blank" class="trailer-btn">▶ Play</a>' if trailer else ''}
        </div>
    </div>
    """, height=260)  # Matches the card height

def main():
    st.set_page_config(layout="wide", page_title="Movie Recommendations", page_icon="🎬")

    # Load data
    movies = load_data()
    if movies is None:
        st.stop()

    # Prepare similarity matrix
    cosine_sim = prepare_model(movies)

    st.markdown("""
    <style>
        .row-container {
            display: flex;
            overflow-x: auto;
            padding: 15px 20px;
            background: #141414;
            margin-bottom: 30px;
            gap: 5px;
        }
        .row-container::-webkit-scrollbar {
            display: none;
        }
        .row-title {
            color: white;
            font-size: 1.3rem;
            font-weight: 500;
            margin: 5px 0 0 20px;
            padding-top: 10px;
        }
        body {
            background-color: #141414;
        }
    </style>
    <div style="background-color: #141414; padding-bottom: 20px;">
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        num_recs = st.slider("Number of recommendations", 3, 10, 5)
        selected = st.selectbox(
            "🎞️ Select a movie you like:",
            movies['title'].sort_values(),
            index=movies['title'].tolist().index("Toy Story (1995)") if "Toy Story (1995)" in movies['title'].values else 0
        )

    if st.button("🔍 Find Similar Movies"):
        with st.spinner("Finding recommendations..."):
            idx = movies[movies['title'] == selected].index[0]
            sim_scores = sorted(
                list(enumerate(cosine_sim[idx])),
                key=lambda x: x[1],
                reverse=True
            )[1:num_recs+1]

            st.markdown(
                f'<div class="row-title">Because you watched: {movies.iloc[idx]["display_title"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="row-container">', unsafe_allow_html=True)
            for sim_idx, _ in sim_scores:
                movie_card(movies.iloc[sim_idx])
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Close outer container


if __name__ == "__main__":
    main()