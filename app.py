import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import os
import zipfile
from io import BytesIO

st.write("## Debug Information")
st.write("Current directory:", os.getcwd())
st.write("Files in .streamlit:", os.listdir(".streamlit"))

try:
    st.write("Secrets contents:", dict(st.secrets))
    st.success("API Key loaded successfully!")
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.warning("Make sure your secrets.toml has [secrets] section header")

# --- Configuration ---
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
DEFAULT_POSTER = "https://via.placeholder.com/150x225?text=No+Poster"
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]  # Set in Streamlit Cloud secrets

# --- Data Loading ---
# Add this at the beginning of your load_data() function
@st.cache_data
def load_data():
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
        
        if not os.path.exists("data/ml-latest-small"):
            st.info("Downloading MovieLens dataset...")
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            response = requests.get(url, timeout=30)
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall("data")
            st.success("Dataset downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading dataset: {e}")
        return None, None
    
    # Rest of your loading code...
    
    # Load movies data
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    
    # Load ratings data
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    
    # Load links to get TMDB IDs
    links = pd.read_csv("data/ml-latest-small/links.csv")
    movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    
    return movies, ratings

movies, ratings = load_data()

# --- TMDB API Functions ---
@st.cache_data(ttl=86400)  # Cache for 1 day
def get_movie_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY}
        )
        data = response.json()
        return f"{POSTER_BASE_URL}{data['poster_path']}" if data.get('poster_path') else None
    except:
        return None

# --- Recommendation Functions ---
@st.cache_data
def prepare_content_model():
    # Create a soup of genres
    movies['soup'] = movies['genres'].apply(lambda x: x.replace('|', ' '))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

@st.cache_data
def prepare_collaborative_model():
    # Create user-item matrix for simple collaborative filtering
    user_item_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    return model, user_item_matrix

cosine_sim = prepare_content_model()
collab_model, user_item_matrix = prepare_collaborative_model()

def content_based_recommendations(movie_id, n=5):
    idx = movies[movies['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

def collaborative_recommendations(movie_id, n=5):
    try:
        distances, indices = collab_model.kneighbors(
            user_item_matrix.loc[movie_id].values.reshape(1, -1), 
            n_neighbors=n+1
        )
        similar_movies = user_item_matrix.iloc[indices[0]].index[1:]  # exclude self
        return movies[movies['movieId'].isin(similar_movies)]
    except:
        return pd.DataFrame()  # Return empty if movie not in training set

def hybrid_recommendations(movie_id, n=5):
    cb_recs = content_based_recommendations(movie_id, n*2)
    cf_recs = collaborative_recommendations(movie_id, n*2)
    
    combined = pd.concat([cb_recs, cf_recs]).drop_duplicates()
    combined['score'] = 0
    combined.loc[combined['movieId'].isin(cb_recs['movieId']), 'score'] += 1
    combined.loc[combined['movieId'].isin(cf_recs['movieId']), 'score'] += 1
    
    return combined.sort_values('score', ascending=False).head(n)

# --- UI Components ---
def movie_card(movie, width=150):
    poster_url = get_movie_poster(movie['tmdbId'])
    with st.container():
        st.image(poster_url if poster_url else DEFAULT_POSTER, width=width)
        st.markdown(f"**{movie['title']}**")
        genres = movie['genres'].split('|')
        st.caption(f"{', '.join(genres[:3])}{'...' if len(genres) > 3 else ''}")

def add_to_history(movie_id):
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    movie_info = movies[movies['movieId'] == movie_id].iloc[0]
    entry = {
        'movieId': movie_id,
        'title': movie_info['title'],
        'genres': movie_info['genres'],
        'tmdbId': movie_info['tmdbId'],
        'timestamp': pd.Timestamp.now()
    }
    
    # Remove if already in history
    st.session_state.history = [x for x in st.session_state.history if x['movieId'] != movie_id]
    st.session_state.history.append(entry)
    
    # Keep only last 5 items
    if len(st.session_state.history) > 5:
        st.session_state.history = st.session_state.history[-5:]

# --- Streamlit App ---
st.set_page_config(
    page_title="Nupoor Mhadgut's Movie Recommender", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .movie-card {
        padding: 10px;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.05);
        background-color: #f0f2f6;
    }
    .stSelectbox:first-child > div:first-child {
        font-weight: bold;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎬 Nupoor Mhadgut's Movie Recommender")
    st.markdown("Using the MovieLens Small Dataset")
    
    rec_type = st.radio(
        "Recommendation Type",
        ["Content-Based", "Collaborative", "Hybrid"],
        index=0
    )
    
    num_recs = st.slider("Number of Recommendations", 3, 10, 5)
    
    st.markdown("---")
    st.subheader("Your Recently Viewed")
    if 'history' in st.session_state and st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.container():
                cols = st.columns([1, 3])
                poster_url = get_movie_poster(item['tmdbId'])
                cols[0].image(poster_url if poster_url else DEFAULT_POSTER, width=60)
                cols[1].write(f"**{item['title']}**")
    else:
        st.write("No history yet")

# Main Content
st.header("Discover Your Next Favorite Movie")

# Movie selection
selected_movie = st.selectbox(
    "Select a movie you like:",
    movies['title'],
    index=100,  # Start with a popular movie
    key="movie_select"
)

if st.button("Get Recommendations", type="primary"):
    selected_movie_id = movies[movies['title'] == selected_movie]['movieId'].values[0]
    add_to_history(selected_movie_id)
    
    with st.spinner(f"Finding similar movies to '{selected_movie}'..."):
        if rec_type == "Content-Based":
            recommendations = content_based_recommendations(selected_movie_id, num_recs)
        elif rec_type == "Collaborative":
            recommendations = collaborative_recommendations(selected_movie_id, num_recs)
        else:
            recommendations = hybrid_recommendations(selected_movie_id, num_recs)
    
    st.subheader(f"Because you liked *{selected_movie}*, you might enjoy:")
    
    # Display recommendations in columns
    cols = st.columns(num_recs)
    for i, (col, (_, movie)) in enumerate(zip(cols, recommendations.iterrows())):
        with col:
            with st.container():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                movie_card(movie)
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    **Dataset**: [MovieLens Small Dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)  
    **API**: [The Movie Database (TMDB)](https://www.themoviedb.org)  
    **Built with**: Streamlit, scikit-learn
""")