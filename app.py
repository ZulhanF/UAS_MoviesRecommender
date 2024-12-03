import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_data(df):

    def clean_text(text):
        if pd.isna(text):
            return ''
        return str(text).strip()
    

    df['Title'] = df['Title'].apply(clean_text)
    df['Overview'] = df['Overview'].apply(clean_text)
    df['Genre'] = df['Genre'].apply(clean_text)
    

    df['Kombinasi'] = df['Title'] + ' ' + df['Overview'] + ' ' + df['Genre']
    

    df = df[df['Kombinasi'].str.strip() != '']
    

    df = df.reset_index(drop=True)
    
    return df

# Fungsi load data dan proses dengan error handling
@st.cache_data
def load_data():
    try:

        df = pd.read_csv('9000plus.csv')
        

        df = preprocess_data(df)
        

        if len(df) == 0:
            st.error("Tidak ada data valid dalam dataset")
            return None, None
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Kombinasi'])
        
        return df, tfidf_matrix
    
    except Exception as e:
        st.error(f"Kesalahan saat memuat data: {e}")
        return None, None


def rekomendasi_film(judul, df, tfidf_matrix, top_n=10):
    try:

        film_cocok = df[df['Title'].str.contains(judul, case=False, na=False)]
        
        if len(film_cocok) == 0:
            st.warning("Tidak ada film yang ditemukan. Coba judul lain.")
            return None
        

        idx = film_cocok.index[0]
        

        similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        

        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        rekomendasi = df.iloc[similar_indices]
        return rekomendasi[['Title', 'Overview', 'Poster_Url', 'Release_Date', 'Genre']]
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses rekomendasi: {e}")
        return None


st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")


df, tfidf_matrix = load_data()


if df is not None and tfidf_matrix is not None:

    all_titles = df['Title'].tolist()
    input_judul = st.selectbox("Pilih atau ketik judul film:", [""] + all_titles, index=0, key="film_select")

    if input_judul:

        rekomendasi = rekomendasi_film(input_judul, df, tfidf_matrix)
        
        if rekomendasi is not None:
            st.subheader(f"Rekomendasi Film Mirip '{input_judul}':")
            
            for _, film in rekomendasi.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:

                        try:
                            st.image(film['Poster_Url'], width=150)
                        except Exception:
                            st.write("Poster tidak tersedia")
                    
                    with col2:
                        st.write(f"**Judul:** {film['Title']}")
                        st.write(f"**Genre:** {film['Genre']}")
                        st.write(f"**Rilis:** {film['Release_Date']}")
                        st.write(f"**Sinopsis:** {film['Overview']}")
                    
                    st.markdown("---")
else:
    st.error("Gagal memuat dataset. Silakan periksa file CSV Anda.")