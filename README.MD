# 🎬 Movie Recommendation App

A Streamlit-powered web app that recommends similar movies and displays their YouTube trailers instantly. Perfect for film enthusiasts!



## ✨ Features

### 1. **Smart Recommendations**
   - **Content-Based Filtering**: Uses TF-IDF and cosine similarity to analyze movie genres.
   - **Accurate Matches**: Recommends movies with similar themes (e.g., "Inception" → "The Matrix").

### 2. **Instant Trailer Playback**
   - **YouTube Integration**: Fetches official trailers via YouTube API.
   - **One-Click Access**: Watch trailers in a new tab with the "Watch Trailer" button.

### 3. **User-Friendly Interface**
   - **Responsive Design**: Works on desktop and mobile.
   - **Interactive Elements**:
     - Dropdown menu with 9,000+ movies.
     - Adjustable number of recommendations (3-10).
   - **Visual Appeal**: Movie posters with clean titles and genres.

### 4. **Performance Optimized**
   - **Caching**: Reduces API calls and load times.
   - **Auto-Downloads**: MovieLens dataset downloads on first run.

---

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- GitHub account
- [YouTube API key](https://developers.google.com/youtube/v3/getting-started)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt

3. **Add your API key**:

   Create .streamlit/secrets.toml and add:

   toml
   [api_keys]
   YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
   OMDB_API_KEY = "your_omdb_key_here"

4. **Run locally**:

   ```bash
   streamlit run app.py
