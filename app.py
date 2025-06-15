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

# --- Constants ---
DEFAULT_THUMBNAIL = "https://via.placeholder.com/300x450?text=No+Poster"
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
            with st.spinner("📦 Downloading MovieLens dataset (this may take a minute)..."):
                response = requests.get(MOVIELENS_URL, timeout=30)
                response.raise_for_status()
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall("data")
        
        movies = pd.read_csv("data/ml-latest-small/movies.csv")
        movies['genres'] = movies['genres'].str.replace('|', ' ')
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        # Clean titles with apostrophes
        movies['display_title'] = movies['title'].str.replace(r"^'", "‘").str.replace(r"'([^s]|s[^ ])", "’\\1")
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
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": f"{title} official trailer",
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
        if data.get('items'):
            return f"https://youtu.be/{data['items'][0]['id']['videoId']}"
    except:
        return None

def fetch_itunes_trailer(title, year=None):
    try:
        query = f"{title} {year}" if year else title
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
            return data["results"][0].get("previewUrl")
    except:
        return None

def get_best_trailer(title, year):
    youtube_url = fetch_youtube_trailer(title)
    if youtube_url:
        return {
            'url': youtube_url,
            'source': 'youtube',
            'button_color': '#FF0000',
            'badge': '🎥 YouTube'
        }
    
    itunes_url = fetch_itunes_trailer(title, year)
    if itunes_url:
        return {
            'url': itunes_url,
            'source': 'itunes',
            'button_color': '#000000',
            'badge': '🍎 iTunes'
        }
    
    return None

def fetch_omdb_poster(title, year):
    if not OMDB_API_KEY:
        return None
    try:
        clean_title = re.sub(r'\([^)]*\)', '', title).strip()
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
        return data.get('Poster') if data.get('Poster') != 'N/A' else None
    except:
        return None

def fetch_itunes_poster(title, year=None):
    try:
        query = f"{title} {year}" if year else title
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
    except:
        return None

@st.cache_data(ttl=3600)
def get_movie_media(movie):
    poster = fetch_omdb_poster(movie['title'], movie.get('year')) if OMDB_API_KEY else None
    if not poster:
        poster = fetch_itunes_poster(movie['title'], movie.get('year'))
    
    return {
        'poster': poster if poster else DEFAULT_THUMBNAIL,
        'trailer': get_best_trailer(movie['title'], movie.get('year'))
    }

# --- Enhanced UI Components ---
def movie_card(movie):
    media = get_movie_media(movie)
    
    with st.container():
        # Card container with shadow and hover effect
        st.markdown(
            """
            <style>
                .movie-card {
                    border-radius: 12px;
                    border: 1px solid #e0e0e0;
                    padding: 16px;
                    margin-bottom: 24px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    transition: transform 0.3s, box-shadow 0.3s;
                    height: 100%;
                    background: white;
                }
                .movie-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 16px rgba(0,0,0,0.12);
                }
            </style>
            <div class="movie-card">
            """,
            unsafe_allow_html=True
        )
        
        # Poster with container width
        st.image(
            media['poster'],
            use_container_width=True,
            output_format="JPEG"
        )
        
        # Movie title with gradient text
        st.markdown(
            f"""
            <h3 style='
                text-align: center;
                margin: 12px 0 8px;
                background: linear-gradient(45deg, #6e48aa, #9d50bb);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.2rem;
                font-weight: 600;
            '>
                {movie['display_title']}
            </h3>
            """,
            unsafe_allow_html=True
        )
        
        # Genres as stylish pills
        genres = movie['genres'].split()[:3]  # Show max 3 genres
        genre_pills = "".join(
            f"""
            <span style='
                background: #f0f2f6;
                border-radius: 16px;
                padding: 4px 12px;
                margin: 4px;
                font-size: 0.75rem;
                display: inline-block;
                color: #555;
            '>{g}</span>
            """ for g in genres
        )
        
        st.markdown(
            f"""
            <div style='text-align: center; margin: 12px 0; line-height: 2;'>
                {genre_pills}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Dynamic trailer button
        if media['trailer']:
            btn_style = f"""
                background: {media['trailer']['button_color']};
                color: white;
                padding: 10px 20px;
                border-radius: 50px;
                text-decoration: none;
                font-weight: 600;
                font-size: 0.9rem;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                transition: all 0.3s;
                width: fit-content;
                margin: 0 auto;
            """
            
            st.markdown(
                f"""
                <div style='text-align: center; margin: 16px 0 8px;'>
                    <a href='{media['trailer']['url']}' target='_blank' style='{btn_style}'>
                        <span>▶ Play Trailer</span>
                        <span style='font-size: 1rem;'>{media['trailer']['badge']}</span>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card

# --- Main App with Professional Header ---
def main():
    st.set_page_config(
        layout="wide",
        page_title="🎬 Nupoor Mhadgut's Movie Recommendation Engine",
        page_icon="🎥"
    )
    
    # Custom header with gradient
    st.markdown(
        """
        <style>
            .header {
                background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
                padding: 3rem 2rem;
                border-radius: 12px;
                margin-bottom: 2.5rem;
                color: white;
            }
            .header h1 {
                font-size: 2.8rem;
                margin: 0;
                text-align: center;
            }
            .header p {
                text-align: center;
                opacity: 0.9;
                margin-top: 0.5rem;
            }
            /* Make selectbox dropdown arrow point down */
            div[data-baseweb="select"] > div:first-child {
                padding-right: 2.5rem;
            }
            div[data-baseweb="select"] > div:first-child > div:after {
                content: "▼";
                position: absolute;
                top: 50%;
                right: 0.5rem;
                transform: translateY(-50%);
                pointer-events: none;
            }
            /* Image container styling */
            .stImage img {
                border-radius: 8px;
                max-width: 100%;
                height: auto;
            }
        </style>
        <div class="header">
            <h1>🎬 Movie Recommendation Engine</h1>
            <p>Discover your next favorite film with AI-powered suggestions</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Main content
    movies = load_data()
    if movies is None:
        st.stop()
    
    cosine_sim = prepare_model(movies)
    
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        num_recs = st.slider("Number of recommendations", 3, 10, 5)
        st.markdown("---")
        st.markdown("Built with ❤️ by [Nupoor Mhadgut]")
    
    # Selectbox with downward-pointing arrow
    selected = st.selectbox(
        "🎞️ Select a movie you like:",
        movies['title'].sort_values(),
        index=movies['title'].tolist().index("Toy Story (1995)") if "Toy Story (1995)" in movies['title'].values else 0,
        help="Start typing to search through 9,000+ movies"
    )
    
    if st.button("🔍 Find Similar Movies", type="primary"):
        with st.spinner("🧠 Analyzing genres and finding recommendations..."):
            idx = movies[movies['title'] == selected].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
            
            st.markdown(f"## 🎯 Similar to: **{movies.iloc[idx]['display_title']}**")
            cols = st.columns(min(3, len(sim_scores)))
            
            for i, (idx, score) in enumerate(sim_scores):
                with cols[i % len(cols)]:
                    movie_card(movies.iloc[idx])

if __name__ == "__main__":
    main()