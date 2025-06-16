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
logging.basicConfig(filename='poster_errors.log', level=logging.INFO)

# Constants
DEFAULT_THUMBNAIL = "https://via.placeholder.com/300x450.png?text=No+Poster"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

if 'api_keys' not in st.secrets:
    st.error("❌ API keys missing in Streamlit secrets!")
    st.stop()

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
    
    # 1. Try OMDB API
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
    
    # 2. Try iTunes API
    try:
        query = f"{clean_title} {year}" if year else clean_title
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
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
    
    # 3. Try Wikipedia fallback
    try:
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={clean_title}&prop=pageimages&format=json&pithumbsize=500"
        response = requests.get(wiki_url, timeout=3)
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        if pages:
            page = next(iter(pages.values()))
            if 'thumbnail' in page:
                return page['thumbnail']['source']
    except Exception as e:
        logging.info(f"Wikipedia failed for {title}: {str(e)}")
    
    return DEFAULT_THUMBNAIL

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_youtube_trailer(title):
    if not YOUTUBE_API_KEY: return None
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
    # Try YouTube first
    youtube_url = fetch_youtube_trailer(title)
    if youtube_url: 
        return {
            'url': youtube_url, 
            'source': 'youtube', 
            'button_color': '#FF0000'
        }
    
    # Fallback to iTunes if YouTube fails
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
    with st.container():
        # Platform logos
        platform_logo = ""
        if media["trailer"]:
            if media["trailer"]["source"] == "youtube":
                platform_logo = """<span style="margin-left:8px;background:url('https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/20px-YouTube_full-color_icon_%282017%29.svg.png') no-repeat; width:20px; height:14px; display:inline-block; vertical-align:middle;"></span>"""
            else:
                platform_logo = """<span style="margin-left:8px;background:url('https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/ITunes_logo.svg/20px-ITunes_logo.svg.png') no-repeat; width:20px; height:14px; display:inline-block; vertical-align:middle;"></span>"""
        
        trailer_html = ""
        if media["trailer"]:
            trailer_html = f"""
            <div style='margin-top:auto;text-align:center;'>
                <a href='{media["trailer"]["url"]}' target='_blank'
                style='display:inline-flex;align-items:center;justify-content:center;background:{media["trailer"]["button_color"]};
                color:white;padding:8px 16px;border-radius:50px;text-decoration:none;font-weight:600;font-size:0.9rem;margin:0 auto;'>
                    ▶{platform_logo}
                </a>
            </div>"""
        
        st.markdown(f"""
        <div style='border-radius:12px;border:1px solid #e0e0e0;padding:16px;margin-bottom:24px;
                    box-shadow:0 4px 12px rgba(0,0,0,0.08);background:white;display:flex;flex-direction:column;height:100%;'>
            <img src='{media["poster"]}' style='width:100%;border-radius:8px;aspect-ratio:2/3;object-fit:contain;background:#f5f5f5;'>
            <h3 style='text-align:center;margin:12px 0 4px;font-size:1.2rem;font-weight:600;color:#333;'>
                {movie['display_title']} ({movie['year']})
            </h3>
            <div style='text-align:center;margin:4px 0 12px;color:#666;font-size:0.9rem;'>
                {', '.join(movie['genres'].split()[:3])}
            </div>
            {trailer_html if media["trailer"] else ""}
        </div>""", unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="🎬 Movie Recommendation Engine", page_icon="🎥")
    st.markdown("""
    <style>
        .header {background:linear-gradient(135deg,#6e48aa 0%,#9d50bb 100%);padding:2rem;border-radius:12px;margin-bottom:2rem;color:white;}
    </style>
    <div class="header">
        <h1 style='text-align:center;margin:0;'>🎬 Movie Recommendation Engine</h1>
    </div>""", unsafe_allow_html=True)
    
    movies = load_data()
    if movies is None: st.stop()
    cosine_sim = prepare_model(movies)
    
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        num_recs = st.slider("Number of recommendations", 3, 10, 5)
    
    selected = st.selectbox(
        "🎞️ Select a movie you like:",
        movies['title'].sort_values(),
        index=movies['title'].tolist().index("Toy Story (1995)") if "Toy Story (1995)" in movies['title'].values else 0
    )
    
    if st.button("🔍 Find Similar Movies", type="primary"):
        with st.spinner("Finding recommendations..."):
            idx = movies[movies['title'] == selected].index[0]
            sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:num_recs+1]
            st.markdown(f"## 🎯 Similar to: **{movies.iloc[idx]['display_title']}**")
            cols = st.columns(min(3, len(sim_scores)))
            for i, (idx, score) in enumerate(sim_scores):
                with cols[i % len(cols)]: movie_card(movies.iloc[idx])

if __name__ == "__main__":
    main()