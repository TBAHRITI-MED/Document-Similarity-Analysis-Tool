import os
import chardet
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

# Télécharger les ressources nécessaires de nltk
nltk.download('punkt')
nltk.download('stopwords')

# Fonction pour détecter l'encodage d'un fichier
def detecter_encodage(filepath):
    with open(filepath, 'rb') as file:
        resultat = chardet.detect(file.read())
        return resultat['encoding']

# Charger et tokenizer un fichier
def charger_et_tokenizer_fichier(file_path):
    try:
        encoding = detecter_encodage(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            texte = file.read()
            phrases = sent_tokenize(texte)
            return texte, phrases
    except Exception as e:
        st.warning(f"Impossible de lire ou de tokenizer le fichier {file_path}: {e}")
        return "", []

# Charger et tokenizer tous les fichiers dans un dossier
def charger_fichiers_et_tokenizer(folder_path):
    documents_tokenized = []
    noms_fichiers = []
    textes_complets = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            texte, phrases = charger_et_tokenizer_fichier(file_path)
            documents_tokenized.extend(phrases)
            textes_complets.append(texte)
            noms_fichiers.append(file_name)
    
    return textes_complets, documents_tokenized, noms_fichiers

# Calcul de la similarité entre une phrase de recherche et une liste de phrases
def calculer_similarite(phrase_recherche, phrases):
    vectorizer = TfidfVectorizer()
    vecteurs = vectorizer.fit_transform([phrase_recherche] + list(phrases))
    similarites = cosine_similarity(vecteurs[0:1], vecteurs[1:]).flatten()
    return similarites

# Fonction pour créer un nuage de mots
def creer_nuage_mots(texte):
    stopwords_fr = set(stopwords.words('french'))
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=stopwords_fr,
        min_font_size=10
    ).generate(texte)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Fonction pour créer un graphique de distribution des scores de similarité
def creer_graphique_distribution(similarites):
    plt.figure(figsize=(10, 5))
    sns.histplot(similarites, bins=20)
    plt.title('Distribution des scores de similarité')
    plt.xlabel('Score de similarité')
    plt.ylabel('Fréquence')
    return plt

# Fonction pour créer un graphique des top N phrases les plus similaires
def creer_graphique_top_phrases(similarites, n=5):
    top_indices = sorted(range(len(similarites)), key=lambda i: similarites[i], reverse=True)[:n]
    top_scores = [similarites[i] for i in top_indices]
    
    plt.figure(figsize=(10, 5))
    plt.barh(range(n), top_scores)
    plt.yticks(range(n), [f'Phrase {i+1}' for i in range(n)])
    plt.xlabel('Score de similarité')
    plt.title(f'Top {n} phrases les plus similaires')
    return plt

# Interface utilisateur avec Streamlit
st.title("Recherche de Documents")
st.write("Ce programme permet de rechercher et d'analyser des documents texte .")

# Choisir entre un fichier ou un dossier
choix_analyse = st.radio("Souhaitez-vous analyser un dossier complet ou un fichier spécifique ?", ("Dossier", "Fichier"))

if choix_analyse == "Dossier":
    folder_path = st.text_input("Choisissez le dossier contenant les fichiers :")

    if folder_path and os.path.isdir(folder_path):
        st.success("Dossier chargé avec succès !")
        textes_complets, documents_tokenized, noms_fichiers = charger_fichiers_et_tokenizer(folder_path)
        st.write(f"Fichiers chargés et tokenisés avec succès ! {len(noms_fichiers)} fichiers trouvés.")
        st.write(f"Nombre total de phrases : {len(documents_tokenized)}")

        # Afficher le nuage de mots global
        st.subheader("Nuage de mots pour tous les documents")
        texte_complet = " ".join(textes_complets)
        st.pyplot(creer_nuage_mots(texte_complet))

        mode_recherche = st.radio("Rechercher dans :", ["Tous les fichiers du dossier", "Un fichier spécifique"])

        sentences = []
        selected_file = None

        if mode_recherche == "Un fichier spécifique" and documents_tokenized:
            selected_file = st.selectbox("Choisissez un fichier :", noms_fichiers)
            if selected_file:
                file_index = noms_fichiers.index(selected_file)
                texte, sentences = charger_et_tokenizer_fichier(os.path.join(folder_path, selected_file))
                
                # Afficher le nuage de mots pour le fichier sélectionné
                st.subheader(f"Nuage de mots pour {selected_file}")
                st.pyplot(creer_nuage_mots(texte))
        else:
            sentences = documents_tokenized

        phrase_recherche = st.text_input("Entrez une phrase pour rechercher dans les documents :")
        k = st.slider("Nombre de résultats les plus similaires à afficher :", 1, 20, 5)

        if phrase_recherche:
            similarites_recherche = calculer_similarite(phrase_recherche, sentences)
            
            # Afficher les visualisations des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution des scores de similarité")
                st.pyplot(creer_graphique_distribution(similarites_recherche))
            
            with col2:
                st.subheader(f"Top {k} phrases les plus similaires")
                st.pyplot(creer_graphique_top_phrases(similarites_recherche, k))
            
            # Afficher les résultats textuels
            indices_similaires = sorted(
                range(len(similarites_recherche)),
                key=lambda i: similarites_recherche[i],
                reverse=True
            )
            
            st.subheader("Résultats détaillés")
            for idx in indices_similaires[:k]:
                st.write(f"Phrase similaire (score : {similarites_recherche[idx]:.4f}):")
                st.write(f"{sentences[idx]}")

elif choix_analyse == "Fichier":
    file_path = st.text_input("Entrez le chemin complet du fichier à analyser :")

    if file_path:
        texte, documents_tokenized = charger_et_tokenizer_fichier(file_path)
        st.write(f"Fichier chargé et tokenisé avec succès !")
        st.write(f"Nombre total de phrases : {len(documents_tokenized)}")

        # Afficher le nuage de mots pour le fichier
        st.subheader("Nuage de mots du document")
        st.pyplot(creer_nuage_mots(texte))

        phrase_recherche = st.text_input("Entrez une phrase pour rechercher dans le fichier :")
        k = st.slider("Nombre de résultats les plus similaires à afficher :", 1, 20, 5)

        if phrase_recherche:
            similarites_recherche = calculer_similarite(phrase_recherche, documents_tokenized)
            
            # Afficher les visualisations des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution des scores de similarité")
                st.pyplot(creer_graphique_distribution(similarites_recherche))
            
            with col2:
                st.subheader(f"Top {k} phrases les plus similaires")
                st.pyplot(creer_graphique_top_phrases(similarites_recherche, k))
            
            # Afficher les résultats textuels
            indices_similaires = sorted(
                range(len(similarites_recherche)),
                key=lambda i: similarites_recherche[i],
                reverse=True
            )
            
            st.subheader("Résultats détaillés")
            for idx in indices_similaires[:k]:
                st.write(f"Phrase similaire (score : {similarites_recherche[idx]:.4f}):")
                st.write(f"{documents_tokenized[idx]}")