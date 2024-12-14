import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np

# Titre de la page
st.title("Chatbot basé sur la similarité")

# Entrée des données initiales
uploaded_file = st.file_uploader("Téléchargez un fichier .txt contenant vos phrases", type="txt")

if uploaded_file:
    # Lire le fichier
    document = uploaded_file.read().decode("utf-8")
    
    # Diviser le texte en phrases
    sentences = sent_tokenize(document)
    st.write("Les phrases détectées dans le texte sont :")
    for idx, sentence in enumerate(sentences, start=1):
        st.write(f"{idx}. {sentence}")

    # Entrée pour la phrase de l'utilisateur
    user_input = st.text_input("Entrez une phrase pour rechercher une correspondance :")

    if user_input:
        # Vectorisation TF-IDF
        vectorizer = TfidfVectorizer()
        corpus = [user_input] + sentences
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Calcul des similarités cosinus
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Trier les phrases similaires
        sorted_indices = np.argsort(similarities)[::-1]
        top_k = st.slider("Nombre de phrases similaires à afficher :", 1, len(sentences), 3)

        st.write("Les phrases les plus similaires sont :")
        for idx in sorted_indices[:top_k]:
            st.write(f"- Similarité {similarities[idx]:.4f}: {sentences[idx]}")

else:
    st.warning("Veuillez téléverser un fichier pour commencer.")

