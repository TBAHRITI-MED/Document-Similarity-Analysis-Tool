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

############################################
# 1) CONFIGURATION STREAMLIT
############################################
st.set_page_config(
    page_title="Recherche de Documents",
    page_icon="📂",
    layout="wide"
)

# Insertion d'un bloc CSS pour le style
st.markdown("""
<style>
    .big-title {
        font-size:250%;
        color: #2F4F4F; /* DarkSlateGray */
        text-align: center;
        margin-top: 0.2em;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size:130%;
        color: #8B008B; /* DarkMagenta */
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .section-heading {
        font-size:115%;
        color: #2E8B57; /* SeaGreen */
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

############################################
# 2) TITRE DE L'APPLICATION
############################################
st.markdown('<h1 class="big-title">📂 Recherche de Documents</h1>', unsafe_allow_html=True)
st.write("Ce programme permet de **rechercher** et d'**analyser** des documents texte (fichiers `.txt`).")

############################################
# 3) FONCTIONS DE CHARGEMENT ET TOKENISATION
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
    charge leur contenu et leurs phrases.

    Retourne :
    - textes_complets : liste[str] (contenu texte intégral de chaque fichier)
    - documents_tokenized : liste[str] (toutes les phrases issues de tous les fichiers)
    - noms_fichiers : liste[str] (noms de fichiers)
    - file_mapping : liste[str] qui indique, pour chaque phrase, le fichier d'où elle provient
    """
    documents_tokenized = []
    noms_fichiers = []
    textes_complets = []
    file_mapping = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            texte, phrases = charger_et_tokenizer_fichier(file_path)
            if texte.strip():
                textes_complets.append(texte)
                noms_fichiers.append(file_name)
                # On enregistre chaque phrase + on note le file_name
                for p in phrases:
                    documents_tokenized.append(p)
                    file_mapping.append(file_name)
    
    return textes_complets, documents_tokenized, noms_fichiers, file_mapping

############################################
# 4) FONCTIONS D'ANALYSE (SIMILARITÉ & VISUELS)
############################################

def calculer_similarite(phrase_recherche, phrases):
    """
    Calcule la similarité cosinus entre `phrase_recherche`
    et chaque phrase de la liste `phrases`, via TF-IDF.
    Retourne un array de similarités (float).
    """
    vectorizer = TfidfVectorizer()
    vecteurs = vectorizer.fit_transform([phrase_recherche] + list(phrases))
    similarites = cosine_similarity(vecteurs[0:1], vecteurs[1:]).flatten()
    return similarites

def creer_nuage_mots(texte):
    """
    Génère un WordCloud (nuage de mots) pour le texte donné,
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

def calculer_similarite_fichiers(textes_complets, index_fichier_reference):
    """
    Compare un fichier (index_fichier_reference) à tous les autres fichiers
    dans la liste textes_complets, via TF-IDF + Cosine Similarity.
    Retourne un tableau de similarités.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textes_complets)
    vecteur_ref = tfidf_matrix[index_fichier_reference]
    similarites = cosine_similarity(vecteur_ref, tfidf_matrix).flatten()
    return similarites

############################################
# 5) INTERFACE UTILISATEUR STREAMLIT
############################################

st.markdown("### Choix du mode d'analyse")
choix_analyse = st.radio(
    "Souhaitez-vous analyser un dossier complet ou un fichier spécifique ?",
    ("Dossier", "Fichier", "Plusieurs dossiers")
)

if choix_analyse == "Plusieurs dossiers":
    st.markdown("Vous avez choisi : **Plusieurs dossiers**")
    st.markdown("[Aller vers multiple_dossiers 🗂️](http://localhost:8504)")
    st.stop()

elif choix_analyse == "Dossier":
    st.markdown("<h3 class='subtitle'>Analyse d'un Dossier</h3>", unsafe_allow_html=True)
    folder_path = st.text_input("Chemin du dossier contenant les fichiers :")

    if folder_path and os.path.isdir(folder_path):
        st.success("Dossier chargé avec succès !")
        textes_complets, documents_tokenized, noms_fichiers, file_mapping = charger_fichiers_et_tokenizer(folder_path)
        st.write(f"Fichiers chargés et tokenisés : **{len(noms_fichiers)}** fichiers trouvés.")
        st.write(f"Nombre total de phrases (tous fichiers confondus) : **{len(documents_tokenized)}**")

        if "show_phrases" not in st.session_state:
            st.session_state.show_phrases = False

        # Bouton pour basculer l'affichage des phrases
        if st.button("📜 Afficher/Cacher toutes les phrases"):
            st.session_state.show_phrases = not st.session_state.show_phrases

        if st.session_state.show_phrases:
            st.markdown("#### Phrases détectées (tous fichiers) :")
            for idx, sentence in enumerate(documents_tokenized, start=1):
                st.write(f"{idx}. {sentence}")
        else:
            st.write("*(Les phrases sont masquées. Cliquez sur le bouton pour les afficher.)*")

        # Nuage de mots global (tous fichiers)
        st.markdown("<h3 class='section-heading'>Nuage de mots (global)</h3>", unsafe_allow_html=True)
        texte_complet = " ".join(textes_complets)
        fig_cloud = creer_nuage_mots(texte_complet)
        st.pyplot(fig_cloud)

        # Choix du type de recherche
        st.markdown("<h3 class='section-heading'>Mode de recherche</h3>", unsafe_allow_html=True)
        mode_recherche = st.radio(
            "Mode de recherche :",
            ["Recherche de phrase", "Recherche dans un fichier spécifique", "Comparer un fichier à d'autres fichiers"]
        )

        if mode_recherche == "Recherche de phrase":
            # RECHERCHE DE PHRASE (TOUS FICHIERS)
            st.markdown("#### Recherche de phrase (TOUS les fichiers)")
            phrase_recherche = st.text_input("Entrez une phrase :")
            k = st.slider("Nombre de résultats à afficher :", 1, 20, 5)

            if phrase_recherche:
                simil_recherche = calculer_similarite(phrase_recherche, documents_tokenized)

                # Graphique distribution + top K
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribution des scores**")
                    plt_dist = creer_graphique_distribution(simil_recherche)
                    st.pyplot(plt_dist)
                with col2:
                    st.markdown(f"**Top {k} phrases**")
                    plt_top = creer_graphique_top_phrases(simil_recherche, k)
                    st.pyplot(plt_top)

                # Résultats textuels
                idxs_sorted = sorted(range(len(simil_recherche)), key=lambda i: simil_recherche[i], reverse=True)
                st.markdown("**Résultats détaillés** :")
                for idx in idxs_sorted[:k]:
                    st.write(f"Phrase similaire (score={simil_recherche[idx]:.4f}) :")
                    nom_fichier_source = file_mapping[idx]
                    st.write(f"- **Fichier** : {nom_fichier_source}")
                    st.write(f"- **Phrase** : {documents_tokenized[idx]}")

        elif mode_recherche == "Recherche dans un fichier spécifique":
            # RECHERCHE DE PHRASE (UN SEUL FICHIER)
            st.markdown("#### Recherche dans un Fichier Spécifique")
            selected_file = st.selectbox("Choisissez un fichier :", noms_fichiers)
            if selected_file:
                file_index = noms_fichiers.index(selected_file)
                texte, sentences = charger_et_tokenizer_fichier(os.path.join(folder_path, selected_file))

                st.markdown(f"**Nuage de mots pour** `{selected_file}` :")
                fig_cloud_local = creer_nuage_mots(texte)
                st.pyplot(fig_cloud_local)

                phrase_recherche = st.text_input(f"Entrez une phrase pour la recherche :")
                k = st.slider("Nombre de résultats :", 1, 20, 5)

                if phrase_recherche:
                    simil_recherche = calculer_similarite(phrase_recherche, sentences)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Distribution des scores**")
                        fig_dist_local = creer_graphique_distribution(simil_recherche)
                        st.pyplot(fig_dist_local)
                    with col2:
                        st.markdown(f"**Top {k} phrases**")
                        fig_top_local = creer_graphique_top_phrases(simil_recherche, k)
                        st.pyplot(fig_top_local)
                    
                    idxs_local = sorted(range(len(simil_recherche)), key=lambda i: simil_recherche[i], reverse=True)
                    st.markdown("**Résultats détaillés** :")
                    for idx in idxs_local[:k]:
                        st.write(f"Phrase similaire (score={simil_recherche[idx]:.4f}) :")
                        nom_fichier_source = file_mapping[idx]
                        st.write(f"- **Fichier** : {nom_fichier_source}")
                        st.write(f"- **Phrase** : {sentences[idx]}")

        else:
            # COMPARER UN FICHIER À D'AUTRES FICHIERS
            st.markdown("#### Comparer un fichier aux autres (dans le dossier)")
            selected_file = st.selectbox("Choisissez un fichier à comparer :", noms_fichiers)
            k = st.slider("Nombre de fichiers similaires à afficher :", 1, 20, 5)

            if selected_file:
                file_index = noms_fichiers.index(selected_file)
                simil_docs = calculer_similarite_fichiers(textes_complets, file_index)

                idxs_sorted = sorted(range(len(simil_docs)), key=lambda i: simil_docs[i], reverse=True)
                st.markdown("**Résultats de similarité** :")
                cpt = 0
                for i in idxs_sorted:
                    if i == file_index:
                        continue
                    cpt += 1
                    st.write(f"Fichier : {noms_fichiers[i]} (score={simil_docs[i]:.4f})")
                    if cpt >= k:
                        break

elif choix_analyse == "Fichier":
    # ANALYSE SUR UN SEUL FICHIER
    st.markdown("<h3 class='subtitle'>Analyse d'un Fichier Unique</h3>", unsafe_allow_html=True)
    file_path = st.text_input("Entrez le chemin complet du fichier à analyser :")

    if file_path:
        texte, documents_tokenized = charger_et_tokenizer_fichier(file_path)
        if texte:
            st.success("Fichier chargé et tokenisé avec succès !")
            st.write(f"Nombre total de phrases : {len(documents_tokenized)}")

            st.markdown("#### Nuage de mots du document")
            fig_cloud_single = creer_nuage_mots(texte)
            st.pyplot(fig_cloud_single)

            phrase_recherche = st.text_input("Entrez une phrase pour rechercher dans ce fichier :")
            k = st.slider("Nombre de résultats les plus similaires à afficher :", 1, 20, 5)

            if phrase_recherche:
                sims_rech = calculer_similarite(phrase_recherche, documents_tokenized)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribution des scores**")
                    fig_dist_sing = creer_graphique_distribution(sims_rech)
                    st.pyplot(fig_dist_sing)
                
                with col2:
                    st.markdown(f"**Top {k} phrases**")
                    fig_top_sing = creer_graphique_top_phrases(sims_rech, k)
                    st.pyplot(fig_top_sing)
                
                idxs_sing = sorted(range(len(sims_rech)), key=lambda i: sims_rech[i], reverse=True)
                st.markdown("**Résultats détaillés** :")
                for idx in idxs_sing[:k]:
                    st.write(f"Phrase similaire (score={sims_rech[idx]:.4f}) :")
                    st.write(f"{documents_tokenized[idx]}")
        else:
            st.error("Le fichier n'a pas pu être chargé ou est vide.")
    else:
        st.info("Veuillez fournir un chemin de fichier texte valide.")

try:
    plt.close('all')
except:
    pass
