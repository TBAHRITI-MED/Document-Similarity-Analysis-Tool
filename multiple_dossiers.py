import os
import chardet
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Téléchargements silencieux NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

###################################
# 1) CONFIGURATION STREAMLIT
###################################
st.set_page_config(
    page_title="Recherche dans plusieurs dossiers",
    page_icon="📂",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 300%;
        text-align: center;
        color: #2E8B57; /* Vert foncé */
        margin-top: 0.2em;
        margin-bottom: 0.2em;
    }
    .subheading {
        font-size: 150%;
        color: #4682B4; /* Bleu acier */
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .section-title {
        font-size: 120%;
        color: #8B008B; /* Violet foncé */
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .small-text {
        font-size: 90%;
        color: #696969; /* Gris foncé */
    }
</style>
""", unsafe_allow_html=True)

###################################
# 2) FONCTIONS DE TOKENISATION
###################################
def detecter_encodage(filepath):
    with open(filepath, 'rb') as file:
        resultat = chardet.detect(file.read())
        return resultat['encoding']

def charger_et_tokenizer_fichier(file_path):
    try:
        encoding = detecter_encodage(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            texte = f.read()
            phrases = sent_tokenize(texte)
            return texte, phrases
    except Exception as e:
        st.warning(f"Impossible de lire/tokenizer {file_path}: {e}")
        return "", []

def charger_fichiers_et_tokenizer(folder_path):
    textes_complets = []
    documents_tokenized = []
    file_mapping = []

    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            texte, phrases = charger_et_tokenizer_fichier(fpath)
            if texte.strip():
                textes_complets.append(texte)
                for p in phrases:
                    documents_tokenized.append(p)
                    file_mapping.append(fname)
    return textes_complets, documents_tokenized, file_mapping

def charger_plusieurs_dossiers(folder_list):
    all_textes = []
    all_docs = []
    all_map = []
    for folder_path in folder_list:
        if os.path.isdir(folder_path):
            t_comp, d_tok, f_map = charger_fichiers_et_tokenizer(folder_path)
            all_textes.extend(t_comp)
            all_docs.extend(d_tok)
            all_map.extend(f_map)
    return all_textes, all_docs, all_map

###################################
# 3) FONCTIONS ANALYSE & VISUELS
###################################
def calculer_similarite(phrase_recherche, phrases):
    vectorizer = TfidfVectorizer()
    vecteurs = vectorizer.fit_transform([phrase_recherche] + phrases)
    sim = cosine_similarity(vecteurs[0:1], vecteurs[1:]).flatten()
    return sim

def creer_nuage_mots(texte):
    stop_fr = set(stopwords.words('french'))
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stop_fr,
        min_font_size=10
    ).generate(texte)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt

def distribution_scores(sim):
    plt.figure(figsize=(10, 5))
    sns.histplot(sim, bins=20)
    plt.title("Distribution des scores de similarité")
    plt.xlabel("Score de similarité")
    plt.ylabel("Fréquence")
    return plt

def top_k_barplot(sim, k=5):
    top_idx = sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)[:k]
    top_scores = [sim[i] for i in top_idx]
    plt.figure(figsize=(10, 5))
    plt.barh(range(k), top_scores)
    plt.yticks(range(k), [f"Phrase {i+1}" for i in range(k)])
    plt.xlabel("Score de similarité")
    plt.title(f"Top {k} phrases")
    return plt

###################################
# 4) APPLICATION STREAMLIT
###################################
st.markdown('<h1 class="main-title">📂 Recherche dans plusieurs dossiers</h1>', unsafe_allow_html=True)

# STATES INIT
if "folders_loaded" not in st.session_state:
    st.session_state["folders_loaded"] = False
if "textes_global" not in st.session_state:
    st.session_state["textes_global"] = []
if "docs_global" not in st.session_state:
    st.session_state["docs_global"] = []
if "map_global" not in st.session_state:
    st.session_state["map_global"] = []
if "show_all_phrases" not in st.session_state:
    st.session_state["show_all_phrases"] = False

#######################################
# SAISIE DES MULTI-DOSSIERS
#######################################
st.markdown('<h2 class="subheading">Charger plusieurs dossiers</h2>', unsafe_allow_html=True)
multi_folder_paths = st.text_area("Entrez les chemins de dossiers (un chemin par ligne) :", height=100)

if st.button("📂 Charger les dossiers"):
    if multi_folder_paths.strip():
        folder_list = [x.strip() for x in multi_folder_paths.split('\n') if x.strip()]
        st.markdown("**Dossiers détectés :**")
        st.write(folder_list)
        
        (textes_complets_global,
         documents_tokenized_global,
         file_mapping_global) = charger_plusieurs_dossiers(folder_list)

        nb_files = len(textes_complets_global)
        nb_phrases = len(documents_tokenized_global)
        st.success(f"{nb_files} fichier(s) chargé(s), {nb_phrases} phrases.")

        # On stocke dans la session
        st.session_state["folders_loaded"] = True
        st.session_state["textes_global"] = textes_complets_global
        st.session_state["docs_global"] = documents_tokenized_global
        st.session_state["map_global"] = file_mapping_global
    else:
        st.warning("❌ Veuillez entrer des chemins de dossiers valides.")

if st.session_state["folders_loaded"]:
    # BOUTON AFFICHER/CACHER TOUTES LES PHRASES
    if st.button("📜 Afficher/Cacher toutes les phrases"):
        st.session_state["show_all_phrases"] = not st.session_state["show_all_phrases"]

    if st.session_state["show_all_phrases"]:
        st.markdown('<h3 class="section-title">Liste de toutes les phrases chargées</h3>', unsafe_allow_html=True)
        for i, phr in enumerate(st.session_state["docs_global"], start=1):
            st.write(f"{i}. **[{st.session_state['map_global'][i-1]}]** : {phr}")
    else:
        st.write("📄 Les phrases chargées sont masquées.")

    # NUAGE DE MOTS
    st.markdown('<h3 class="section-title">Nuage de mots (multi-dossiers)</h3>', unsafe_allow_html=True)
    texte_concat = " ".join(st.session_state["textes_global"])
    if texte_concat.strip():
        fig_wc = creer_nuage_mots(texte_concat)
        st.pyplot(fig_wc)
    else:
        st.write("Aucun texte disponible pour générer un nuage de mots.")

    # FORMULAIRE DE RECHERCHE
    st.markdown('<h2 class="subheading">Recherche de phrase (multi-dossiers)</h2>', unsafe_allow_html=True)
    with st.form("form_recherche_multi"):
        phrase_recherche = st.text_input("Entrez une phrase à rechercher :")
        k = st.slider("Nombre de résultats :", 1, 20, 5)
        bouton_rechercher = st.form_submit_button("🔍 Rechercher")

    if bouton_rechercher:
        if phrase_recherche.strip():
            sim = calculer_similarite(phrase_recherche, st.session_state["docs_global"])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribution des scores")
                fig_dist = distribution_scores(sim)
                st.pyplot(fig_dist)

            with col2:
                st.subheader(f"Top {k} phrases")
                fig_top = top_k_barplot(sim, k)
                st.pyplot(fig_top)

            idx_sorted = sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)
            st.markdown("### Résultats de la recherche")
            for idx_s in idx_sorted[:k]:
                score_val = sim[idx_s]
                fichier_source = st.session_state["map_global"][idx_s]
                phrase_source = st.session_state["docs_global"][idx_s]
                st.write(f"**Score** : {score_val:.4f}")
                st.write(f"**Fichier** : {fichier_source}")
                st.write(f"**Phrase** : {phrase_source}")
                st.write("---")
        else:
            st.warning("❌ Veuillez entrer une phrase non vide.")
else:
    st.info("ℹ️ Aucun dossier n'est encore chargé.")
