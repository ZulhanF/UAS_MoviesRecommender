import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi load data dan proses
@st.cache_data
def load_data():
    df = pd.read_csv('movies_dataset.csv')
    df['Kombinasi'] = df['Title'] + ' ' + df['Overview'] + ' ' + df['Genre']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Kombinasi'])
    
    return df, tfidf_matrix

# Fungsi rekomendasi film
def rekomendasi_film(judul, df, tfidf_matrix, top_n=5):
    # Filter film yang mirip judulnya
    film_cocok = df[df['Title'].str.contains(judul, case=False)]
    
    if len(film_cocok) == 0:
        st.warning("Tidak ada film yang ditemukan. Coba judul lain.")
        return None
    
    # Ambil indeks film pertama
    idx = film_cocok.index[0]
    
    # Hitung similarity
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Dapatkan top-n rekomendasi
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    rekomendasi = df.iloc[similar_indices]
    return rekomendasi[['Title', 'Overview', 'Poster_Url', 'Release_Date', 'Genre']]

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

# Load data
df, tfidf_matrix = load_data()

# Autocomplete input
all_titles = df['Title'].tolist()
input_judul = st.text_input("Masukkan judul film:", key="film_input")
input_judul = st.selectbox("Pilih atau ketik judul film:", [""] + all_titles, index=0, key="film_select")

if input_judul:
    # Tampilkan rekomendasi
    rekomendasi = rekomendasi_film(input_judul, df, tfidf_matrix)
    
    if rekomendasi is not None:
        st.subheader(f"Rekomendasi Film Mirip '{input_judul}':")
        
        for _, film in rekomendasi.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.image(film['Poster_Url'], width=150)
                
                with col2:
                    st.write(f"**Judul:** {film['Title']}")
                    st.write(f"**Genre:** {film['Genre']}")
                    st.write(f"**Rilis:** {film['Release_Date']}")
                    st.write(f"**Sinopsis:** {film['Overview']}")
                
                st.markdown("---")

# Requirement dependencies
# streamlit
# pandas
# scikit-learn
# numpy