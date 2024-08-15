import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from APi_key import API_KEY

@st.cache_data
# تحميل البيانات
def load_data():
    
    movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    return movies.merge(credits, on='title')

@st.cache_data
# معالجة البيانات
def preprocess_data(movies):
    columns_to_remove = [
        'budget', 'vote_count', 'homepage', 'original_language',
        'original_title', 'popularity', 'production_companies',
        'production_countries', 'release_date', 'revenue', 'runtime',
        'spoken_languages', 'status', 'tagline', 'vote_average', 'id'
    ]
    movies.drop(columns=columns_to_remove, inplace=True)
    movies.dropna(inplace=True)

    # تحويل الأعمدة المعقدة إلى قوائم
    movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
    movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

    # تنظيف القوائم
    for col in ['cast', 'crew', 'genres', 'keywords']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # إعداد العلامات للتوصيات
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    # تصغير العلامات
    ps = PorterStemmer()
    movies['tags'] = movies['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

    return movies

# حساب التشابه
def calculate_similarity(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

@st.cache_data
# وظيفة البحث
def search_movies(query=None, year=None, genre=None, actor=None, min_rating=None):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": API_KEY,
        "query": query,
        "year": year,
        "language": "en"
    }
    if genre:
        params['with_genres'] = genre
    if actor:
        params['with_cast'] = actor
    if min_rating:
        params['vote_average.gte'] = min_rating
    response = requests.get(url, params=params)
    return response.json().get("results", [])

# وظيفة التوصية
def recommend(movie_title, movies, similarity):
    index = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    return [movies.iloc[i[0]].title for i in distances[1:6]]

# التطبيق الرئيسي
def main():
    st.set_page_config(page_title='Movie Recommendation System', page_icon='🎥', layout='wide')
    st.title('🎬 Movie Recommendation and Search System')
    
    # صورة البانر
    st.image("view-3d-cinema-film-reel.jpg", width=800, use_column_width=False)
    st.markdown("""
        This application helps you discover movies based on your interests. 
        You can search for movies by title, genre, or actor. 
        Additionally, you can get recommendations based on your selected movie.
    """)

    # تحميل ومعالجة البيانات
    movies = load_data()
    movies = preprocess_data(movies)
    similarity = calculate_similarity(movies)

    # الشريط الجانبي للبحث
    st.sidebar.header("🔍 Search Options")
    search_type = st.sidebar.selectbox("Select Search Type:", ["By Title", "By Genre", "By Actor"])

    # إدخال المستخدم للبحث
    movie_title = st.sidebar.text_input("Enter Movie Title:")
    release_year = st.sidebar.text_input("Enter Release Year:")
    genre = st.sidebar.text_input("Enter Genre (comma-separated):")
    actor = st.sidebar.text_input("Enter Actor (comma-separated):")
    min_rating = st.sidebar.slider("Minimum Rating:", 0.0, 10.0, 0.0)

    # زر البحث
    if st.sidebar.button("🔍 Search"):
        with st.spinner("Searching..."):
            recommendations = search_movies(movie_title, release_year, genre, actor, min_rating)
            if recommendations:
                st.subheader("🎥 Movie Search Results:")
                cols = st.columns(3)

                for i, movie in enumerate(recommendations):
                    with cols[i % 3]:
                        st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}", width=150)
                        st.write(f"**Title:** {movie['title']}")
                        st.write(f"**Release Date:** {movie['release_date']}")
                        st.write(f"**Rating:** {movie['vote_average']}/10")
                        st.write(f"**Overview:** {movie['overview'][:100]}...")
                        st.write("---")
            else:
                st.warning("❌ No movies found. Please try different search criteria.")

    # وظيفة التوصية
    selected_movie = st.text_input('🔄 Enter the title of a movie for recommendations:')

    if st.button('Recommend'):
        if selected_movie:
            recommendations = recommend(selected_movie, movies, similarity)
            st.subheader('🌟 Recommended Movies:')
            for movie in recommendations:
                st.write(movie)
        else:
            st.warning("⚠️ Please enter a movie title for recommendations.")

if __name__ == "__main__":
    main()
