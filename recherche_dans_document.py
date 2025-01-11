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

# Téléchargements NLTK silencieux
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# nltk.download('punkt_tab', quiet=True)  # Si nécessaire, généralement 'punkt' suffit

############################################
# FONCTIONS DE CHARGEMENT ET DE TOKENISATION
############################################

def detecter_encodage(filepath):
    """Détecte l'encodage du fichier."""
    with open(filepath, 'rb') as file:
        resultat = chardet.detect(file.read())
        return resultat['encoding']

def charger_et_tokenizer_fichier(file_path):
    """
    Ouvre un fichier texte, détecte l'encodage, le lit,
    et le découpe en phrases (sent_tokenize).
    Retourne (texte_complet, liste_de_phrases).
    """
    try:
        encoding = detecter_encodage(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            texte = file.read()
            phrases = sent_tokenize(texte)
            return texte, phrases
    except Exception as e:
        st.warning(f"Impossible de lire ou de tokenizer le fichier {file_path}: {e}")
        return "", []

def charger_fichiers_et_tokenizer(folder_path):
    """
    Parcourt tous les .txt dans un dossier,
    charge leur contenu et leurs phrases,
    retourne :
    - textes_complets : liste de strings (le texte intégral de chaque fichier)
    - documents_tokenized : liste de toutes les phrases de tous les fichiers
    - noms_fichiers : liste des noms de fichiers
    """
    documents_tokenized = []
    noms_fichiers = []
    textes_complets = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            texte, phrases = charger_et_tokenizer_fichier(file_path)
            documents_tokenized.extend(phrases)  # on ajoute toutes les phrases dans la liste "globale"
            textes_complets.append(texte)        # on garde le texte complet du fichier
            noms_fichiers.append(file_name)      # on note le nom du fichier
    
    return textes_complets, documents_tokenized, noms_fichiers

############################################
# FONCTIONS D'ANALYSE (SIMILARITÉ & VISUELS)
############################################

def calculer_similarite(phrase_recherche, phrases):
    """
    Calcule la similarité cosinus entre `phrase_recherche`
    et chaque phrase de la liste `phrases`, via TF-IDF.
    Retourne un tableau de similarités (float).
    """
    vectorizer = TfidfVectorizer()
    vecteurs = vectorizer.fit_transform([phrase_recherche] + list(phrases))
    similarites = cosine_similarity(vecteurs[0:1], vecteurs[1:]).flatten()
    return similarites

def creer_nuage_mots(texte):
    """
    Crée et retourne un WordCloud (nuage de mots) pour le texte donné,
    en filtrant les stopwords français.
    """
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

def creer_graphique_distribution(similarites):
    """
    Crée un histogramme de distribution des scores de similarité.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(similarites, bins=20)
    plt.title('Distribution des scores de similarité')
    plt.xlabel('Score de similarité')
    plt.ylabel('Fréquence')
    return plt

def creer_graphique_top_phrases(similarites, n=5):
    """
    Crée un bar plot pour afficher les top N phrases les plus similaires.
    """
    top_indices = sorted(range(len(similarites)), key=lambda i: similarites[i], reverse=True)[:n]
    top_scores = [similarites[i] for i in top_indices]
    
    plt.figure(figsize=(10, 5))
    plt.barh(range(n), top_scores)
    plt.yticks(range(n), [f'Phrase {i+1}' for i in range(n)])
    plt.xlabel('Score de similarité')
    plt.title(f'Top {n} phrases les plus similaires')
    return plt

###############################
# NOUVELLE FONCTION :
# COMPARAISON FICHIER À FICHIERS
###############################

def calculer_similarite_fichiers(textes_complets, index_fichier_reference):
    """
    Compare un fichier (index_fichier_reference) à tous les autres fichiers
    de la liste textes_complets, via TF-IDF.
    Retourne un array de similarités (une valeur pour chaque fichier).
    """
    vectorizer = TfidfVectorizer()
    # On vectorise tous les documents (fichiers)
    tfidf_matrix = vectorizer.fit_transform(textes_complets)
    # On récupère le vecteur du fichier cible
    vecteur_ref = tfidf_matrix[index_fichier_reference]
    # Calcul de la similarité cosinus avec tous les fichiers
    similarites = cosine_similarity(vecteur_ref, tfidf_matrix).flatten()
    return similarites

############################################
# INTERFACE UTILISATEUR AVEC STREAMLIT
############################################

st.title("Recherche de Documents")
st.write("Ce programme permet de rechercher et d'analyser des documents texte.")

# Choisir entre un fichier ou un dossier
choix_analyse = st.radio(
    "Souhaitez-vous analyser un dossier complet ou un fichier spécifique ?",
    ("Dossier", "Fichier")
)

if choix_analyse == "Dossier":
    folder_path = st.text_input("Chemin du dossier contenant les fichiers :")

    if folder_path and os.path.isdir(folder_path):
        st.success("Dossier chargé avec succès !")
        textes_complets, documents_tokenized, noms_fichiers = charger_fichiers_et_tokenizer(folder_path)
        st.write(f"Fichiers chargés et tokenisés : {len(noms_fichiers)} fichiers trouvés.")
        st.write(f"Nombre total de phrases (tous fichiers confondus) : {len(documents_tokenized)}")

        if "show_phrases" not in st.session_state:
            st.session_state.show_phrases = False

        # Bouton pour basculer l'affichage des phrases
        if st.button("Afficher/Cacher toutes les phrases"):
            st.session_state.show_phrases = not st.session_state.show_phrases

        if st.session_state.show_phrases:
            st.write("Les phrases détectées (tous fichiers) :")
            for idx, sentence in enumerate(documents_tokenized, start=1):
                st.write(f"{idx}. {sentence}")
        else:
            st.write("Les phrases sont masquées. Cliquez sur le bouton pour les afficher.")

        # Nuage de mots global (tous fichiers)
        st.subheader("Nuage de mots pour tous les documents")
        texte_complet = " ".join(textes_complets)
        st.pyplot(creer_nuage_mots(texte_complet))

        # Choix du type de recherche
        mode_recherche = st.radio(
            "Mode de recherche :",
            ["Recherche de phrase", "Recherche dans un fichier spécifique", "Comparer un fichier à d'autres fichiers"]
        )

        if mode_recherche == "Recherche de phrase":
            ############################################
            # RECHERCHE DE PHRASE (TOUS FICHIERS)
            ############################################
            phrase_recherche = st.text_input("Entrez une phrase à rechercher dans TOUS les fichiers :")
            k = st.slider("Nombre de résultats les plus similaires à afficher :", 1, 20, 5)

            if phrase_recherche:
                similarites_recherche = calculer_similarite(phrase_recherche, documents_tokenized)
                
                # Graphique distribution + top K
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribution des scores de similarité")
                    st.pyplot(creer_graphique_distribution(similarites_recherche))
                with col2:
                    st.subheader(f"Top {k} phrases les plus similaires")
                    st.pyplot(creer_graphique_top_phrases(similarites_recherche, k))
                
                # Résultats textuels
                indices_similaires = sorted(
                    range(len(similarites_recherche)),
                    key=lambda i: similarites_recherche[i],
                    reverse=True
                )
                
                st.subheader("Résultats détaillés")
                for idx in indices_similaires[:k]:
                    st.write(f"Phrase similaire (score : {similarites_recherche[idx]:.4f}):")
                    st.write(f"{documents_tokenized[idx]}")

        elif mode_recherche == "Recherche dans un fichier spécifique":
            ############################################
            # RECHERCHE DE PHRASE (UN SEUL FICHIER)
            ############################################
            selected_file = st.selectbox("Choisissez un fichier :", noms_fichiers)
            if selected_file:
                file_index = noms_fichiers.index(selected_file)
                texte, sentences = charger_et_tokenizer_fichier(os.path.join(folder_path, selected_file))
                
                # Nuage de mots pour le fichier choisi
                st.subheader(f"Nuage de mots pour {selected_file}")
                st.pyplot(creer_nuage_mots(texte))

                phrase_recherche = st.text_input(f"Entrez une phrase pour rechercher dans {selected_file} :")
                k = st.slider("Nombre de résultats les plus similaires à afficher :", 1, 20, 5)

                if phrase_recherche:
                    similarites_recherche = calculer_similarite(phrase_recherche, sentences)
                    
                    # Graphique distribution + top K
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Distribution des scores de similarité")
                        st.pyplot(creer_graphique_distribution(similarites_recherche))
                    with col2:
                        st.subheader(f"Top {k} phrases les plus similaires")
                        st.pyplot(creer_graphique_top_phrases(similarites_recherche, k))
                    
                    # Résultats textuels
                    indices_similaires = sorted(
                        range(len(similarites_recherche)),
                        key=lambda i: similarites_recherche[i],
                        reverse=True
                    )
                    
                    st.subheader("Résultats détaillés")
                    for idx in indices_similaires[:k]:
                        st.write(f"Phrase similaire (score : {similarites_recherche[idx]:.4f}):")
                        st.write(f"{sentences[idx]}")

        else:
            ############################################
            # COMPARER UN FICHIER À D'AUTRES FICHIERS
            ############################################
            st.subheader("Comparer un fichier à tous les autres dans le dossier")
            selected_file = st.selectbox("Choisissez un fichier à comparer :", noms_fichiers)
            k = st.slider("Nombre de fichiers les plus similaires à afficher :", 1, 20, 5)

            if selected_file:
                # Récupérer l'index du fichier choisi
                file_index = noms_fichiers.index(selected_file)
                # Calculer la similarité du fichier avec tous les fichiers
                similarites_docs = calculer_similarite_fichiers(textes_complets, file_index)

                # On veut trier les fichiers par similarité
                # NB: similarites_docs[file_index] = 1.0 (c'est le même fichier)
                indices_tries = sorted(range(len(similarites_docs)),
                                       key=lambda i: similarites_docs[i],
                                       reverse=True)
                
                st.subheader("Résultats de similarité entre fichiers")
                cpt = 0
                for i in indices_tries:
                    if i == file_index:
                        continue  # on saute le fichier lui-même
                    cpt += 1
                    st.write(f"Fichier : {noms_fichiers[i]} (score={similarites_docs[i]:.4f})")
                    if cpt >= k:
                        break

elif choix_analyse == "Fichier":
    ############################################
    # ANALYSE SUR UN SEUL FICHIER (CHEMIN TEXTE)
    ############################################
    file_path = st.text_input("Entrez le chemin complet du fichier à analyser :")

    if file_path:
        texte, documents_tokenized = charger_et_tokenizer_fichier(file_path)
        if texte:
            st.write(f"Fichier chargé et tokenisé avec succès !")
            st.write(f"Nombre total de phrases : {len(documents_tokenized)}")

            # Nuage de mots pour le fichier
            st.subheader("Nuage de mots du document")
            st.pyplot(creer_nuage_mots(texte))

            phrase_recherche = st.text_input("Entrez une phrase pour rechercher dans ce fichier :")
            k = st.slider("Nombre de résultats les plus similaires à afficher :", 1, 20, 5)

            if phrase_recherche:
                similarites_recherche = calculer_similarite(phrase_recherche, documents_tokenized)
                
                # Visualisations
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribution des scores de similarité")
                    st.pyplot(creer_graphique_distribution(similarites_recherche))
                
                with col2:
                    st.subheader(f"Top {k} phrases les plus similaires")
                    st.pyplot(creer_graphique_top_phrases(similarites_recherche, k))
                
                # Résultats textuels
                indices_similaires = sorted(
                    range(len(similarites_recherche)),
                    key=lambda i: similarites_recherche[i],
                    reverse=True
                )
                
                st.subheader("Résultats détaillés")
                for idx in indices_similaires[:k]:
                    st.write(f"Phrase similaire (score : {similarites_recherche[idx]:.4f}):")
                    st.write(f"{documents_tokenized[idx]}")
        else:
            st.error("Le fichier n'a pas pu être chargé ou est vide.")
