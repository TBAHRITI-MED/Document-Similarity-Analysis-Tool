import streamlit as st
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, braycurtis
from scipy.special import kl_div
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import webbrowser
import nltk
from scipy.spatial.distance import cdist
import gensim.downloader as api
from gensim.models import Word2Vec, FastText
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Téléchargements NLTK silencieux
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

############################################
# 1) CONFIGURATION STREAMLIT
############################################
st.set_page_config(
    page_title="Analyse de similarité de documents",
    page_icon="📄",
    layout="wide"
)

# Un peu de style CSS
st.markdown("""
<style>
    /* Titre principal plus grand */
    .title-custom {
        font-size: 250%;
        color: #4B0082; /* Indigo */
    }
    /* Sous-titres */
    .subtitle-custom {
        font-size: 150%;
        color: #2F4F4F; /* DarkSlateGray */
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    /* Section headings */
    .section-heading {
        font-size: 120%;
        color: #2E8B57; /* SeaGreen */
        margin-top: 0.8em;
        margin-bottom: 0.3em;
    }
</style>
""", unsafe_allow_html=True)

############################################
# 2) FONCTIONS DE CHARGEMENT ET TOKENISATION
############################################

def preprocess_text(sentences, remove_stopwords, apply_stemming, selected_stemmer_name, langue):
    """
    Prétraitement : tokenisation, suppression stop words, stemming (optionnel).
    """
    stop_words = set(stopwords.words('french' if langue == "Français" else 'english'))
    
    # Sélection du stemmer
    if selected_stemmer_name == "Porter":
        stemmer = PorterStemmer()
    elif selected_stemmer_name == "Lancaster":
        stemmer = LancasterStemmer()
    elif selected_stemmer_name == "Snowball (English)":
        stemmer = SnowballStemmer("english")
    elif selected_stemmer_name == "Snowball (French)":
        stemmer = SnowballStemmer("french")
    else:
        stemmer = None  # Pas de stemming si non valide
    
    processed_sentences = []
    for sentence in sentences:
        # Tokenisation
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        
        # Stop words
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        
        # Stemming
        if apply_stemming and stemmer:
            tokens = [stemmer.stem(word) for word in tokens]
                
        processed_sentences.append(" ".join(tokens))
    
    # Extraction de tous les tokens uniques
    unique_tokens = set()
    for sentence in processed_sentences:
        unique_tokens.update(sentence.split())
    
    return sorted(unique_tokens), processed_sentences

def split_into_sentences(text):
    """
    Divise le texte en phrases selon la ponctuation (.!?).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences

def create_matrices(sentences, unique_tokens, normalization_type):
    """
    Crée la matrice binaire et la matrice d'occurrences (event. normalisée).
    """
    binary_matrix = []
    occurrence_matrix = []
    
    for sentence in sentences:
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        binary_row = [1 if token in tokens else 0 for token in unique_tokens]
        binary_matrix.append(binary_row)
        occurrence_row = [tokens.count(token) for token in unique_tokens]
        occurrence_matrix.append(occurrence_row)

    binary_matrix = np.array(binary_matrix)
    occurrence_matrix = np.array(occurrence_matrix)

    # Normalisation
    if normalization_type == "Probabilité":
        occurrence_matrix = occurrence_matrix / occurrence_matrix.sum(axis=1, keepdims=True)
    elif normalization_type == "L2":
        occurrence_matrix = normalize(occurrence_matrix, norm='l2')

    return binary_matrix, occurrence_matrix

############################################
# 3) FONCTIONS DE DISTANCE / SIMILARITÉ
############################################

def calculate_manhattan_distance(matrix):
    return squareform(pdist(matrix, metric='cityblock'))

def calculate_euclidean_distance(matrix):
    return squareform(pdist(matrix, metric='euclidean'))

def calculate_jaccard_distance(binary_matrix):
    jaccard_distances = pdist(binary_matrix, metric='jaccard')
    return squareform(jaccard_distances)

def calculate_hamming_distance(binary_matrix):
    hamming_distances = pdist(binary_matrix, metric='hamming')
    return squareform(hamming_distances)

def calculate_bray_curtis_distance(matrix):
    bray_curtis_distances = pdist(matrix, metric=braycurtis)
    return squareform(bray_curtis_distances)

def calculate_kl_divergence(p, q):
    return np.sum(kl_div(p, q))

def calculate_kullback_leibler_distance(matrix):
    matrix = np.clip(matrix, 1e-10, None)
    kl_distances = squareform(pdist(matrix, metric=lambda u, v: np.sum(kl_div(u, v))))
    return kl_distances

def calculate_cosine_distance(matrix):
    """Calcule la distance de Cosinus."""
    return squareform(pdist(matrix, metric='cosine'))

def calculate_similarity_matrix(distance_matrix):
    max_distance = np.max(distance_matrix)
    return 1 - (distance_matrix / max_distance)

def K_plus_proches_documents(doc_requete, k, similarity_matrix, sentences):
    similarites = similarity_matrix[doc_requete]
    similarites_idx = [(i, similarites[i]) for i in range(len(similarites)) if i != doc_requete]
    similarites_idx.sort(key=lambda x: x[1], reverse=True)
    return [(idx, similarity, sentences[idx]) for idx, similarity in similarites_idx[:k]]

############################################
# 4) VISUALISATION / WORDCLOUD
############################################

def creer_nuage_mots(texte, langue="Français", remove_stopwords=True, background_color='white'):
    if remove_stopwords:
        stopwords_fr = set(stopwords.words('french' if langue == "Français" else 'english'))
    else:
        stopwords_fr = None

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        stopwords=stopwords_fr,
        max_words=150
    ).generate(texte)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt


############################################
# 5) EMBEDDING Word2Vec / FastText
############################################

def preprocess_for_embedding(sentences):
    return [sentence.lower().split() for sentence in sentences]

def train_word2vec(processed_sentences):
    model = Word2Vec(
        sentences=processed_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    return model

def train_fasttext(processed_sentences):
    model = FastText(
        sentences=processed_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    return model

def get_sentence_vector(model, sentence_words):
    vectors = []
    for word in sentence_words:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            continue
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

############################################
# 6) APPLICATION
############################################
st.markdown('<h1 class="title-custom">Analyse de similarité de documents</h1>', unsafe_allow_html=True)

st.sidebar.subheader("Paramètres de configuration 🛠️")
st.sidebar.write("### Navigation :")
page_selection = st.sidebar.radio("Sélectionner une page :", ["Page principale", "Recherche dans un document", "Chatbot"])

if page_selection == "Recherche dans un document":
    webbrowser.open("http://localhost:8502/")
    st.stop()
elif page_selection == "Chatbot":
    webbrowser.open("http://localhost:8503/")
    st.stop()
else:
    st.sidebar.write("""
    **Instructions :** Sélectionnez les options ci-dessus pour configurer l'analyse de similarité.
    """)

    langue = st.sidebar.radio("Choisissez la langue du texte :", ("Français", "Anglais"))

    st.sidebar.subheader("Prétraitement du texte ✂️")
    remove_stopwords = st.sidebar.checkbox("Supprimer les stop words", value=True)
    apply_stemming = st.sidebar.checkbox("Appliquer le stemming", value=False)

    # Filtrage des stemmers selon la langue
    if langue == "Français":
        available_stemmers = {
            "Lancaster": LancasterStemmer(),
            "Snowball (French)": SnowballStemmer("french")
        }
    else:
        available_stemmers = {
            "Porter": PorterStemmer(),
            "Lancaster": LancasterStemmer(),
            "Snowball (English)": SnowballStemmer("english")
        }
    selected_stemmer_name = st.sidebar.selectbox("Choisissez le type de stemming :", list(available_stemmers.keys()))
    selected_stemmer = available_stemmers[selected_stemmer_name]

    descripteur = st.sidebar.selectbox("Choisissez le descripteur à utiliser :", ["Binaire", "Occurrence", "TF-IDF"])
    normalization_type = st.sidebar.selectbox("Choisissez la méthode de normalisation :", ["Aucune", "Probabilité", "L2"])

    distance_type = st.sidebar.selectbox(
        "Choisissez la métrique de distance :",
        ["Manhattan", "Euclidienne", "Jaccard", "Hamming", "Bray-Curtis", "Kullback-Leibler", "Cosinus"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres d'embedding ✨")
    embedding_type = st.sidebar.selectbox("Choisissez le type d'embedding", ["Word2Vec", "FastText", "Aucun"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Nuage de mots ☁️")
    show_wordcloud = st.sidebar.checkbox("Afficher le nuage de mots")
    wordcloud_bg_color = st.sidebar.color_picker("Couleur de fond du nuage de mots", "#ffffff")
    wordcloud_max_words = st.sidebar.slider("Nombre maximum de mots", 50, 300, 150)

    input_mode = st.sidebar.radio("Comment voulez-vous entrer le texte ?", ("Rédaction manuelle", "Déposer un fichier .txt"))

    if input_mode == "Rédaction manuelle":
        chiraq_text = st.text_area("📝 Rédigez ou collez votre texte ici :")
    else:
        uploaded_file = st.file_uploader("Déposez votre fichier texte ici", type="txt")
        if uploaded_file is not None:
            chiraq_text = uploaded_file.read().decode("utf-8")
        else:
            chiraq_text = ""

############################################
# 7) TRAITEMENT ET ANALYSE
############################################
if chiraq_text:
    with st.spinner("Traitement en cours..."):
        sentences = split_into_sentences(chiraq_text)
        unique_tokens, processed_sentences = preprocess_text(
            sentences,
            remove_stopwords,
            apply_stemming,
            selected_stemmer_name,
            langue
        )
        binary_matrix, occurrence_matrix = create_matrices(processed_sentences, unique_tokens, normalization_type)

    if descripteur == "Binaire":
        matrix = binary_matrix
    elif descripteur == "Occurrence":
        matrix = occurrence_matrix
    elif descripteur == "TF-IDF":
        vectorizer_tfidf = TfidfVectorizer()
        corpus_for_tfidf = [" ".join(s.split()) for s in processed_sentences]
        tfidf_matrix = vectorizer_tfidf.fit_transform(corpus_for_tfidf)
        matrix = tfidf_matrix.toarray()

    if distance_type == "Manhattan":
        distance_matrix = calculate_manhattan_distance(matrix)
    elif distance_type == "Euclidienne":
        distance_matrix = calculate_euclidean_distance(matrix)
    elif distance_type == "Jaccard":
        distance_matrix = calculate_jaccard_distance(binary_matrix)
    elif distance_type == "Hamming":
        distance_matrix = calculate_hamming_distance(binary_matrix)
    elif distance_type == "Bray-Curtis":
        distance_matrix = calculate_bray_curtis_distance(matrix)
    elif distance_type == "Kullback-Leibler":
        distance_matrix = calculate_kullback_leibler_distance(matrix)
    elif distance_type == "Cosinus":
        distance_matrix = calculate_cosine_distance(matrix)

    distance_df = pd.DataFrame(
        distance_matrix,
        columns=[f'Doc {i+1}' for i in range(len(sentences))],
        index=[f'Doc {i+1}' for i in range(len(sentences))]
    )
    st.markdown('<h2 class="subtitle-custom">Matrice de distance</h2>', unsafe_allow_html=True)
    st.dataframe(distance_df)

    similarity_matrix = 1 - (distance_matrix / np.max(distance_matrix))
    similarity_df = pd.DataFrame(
        similarity_matrix,
        columns=[f'Doc {i+1}' for i in range(len(sentences))],
        index=[f'Doc {i+1}' for i in range(len(sentences))]
    )
    st.markdown('<h2 class="subtitle-custom">Matrice de similarité</h2>', unsafe_allow_html=True)
    st.dataframe(similarity_df)

    options_docs = [
        f"Document {i + 1}: {processed_sentences[i][:100]}..."
        if len(processed_sentences[i]) > 100
        else f"Document {i + 1}: {processed_sentences[i]}"
        for i in range(len(processed_sentences))
    ]
    st.write(options_docs)

    if show_wordcloud:
     st.markdown('<h2 class="section-heading">Nuage de mots</h2>', unsafe_allow_html=True)
     wc_plot = creer_nuage_mots(chiraq_text, langue=langue, remove_stopwords=remove_stopwords, background_color=wordcloud_bg_color)
     st.pyplot(wc_plot)


    if embedding_type != "Aucun":
        st.markdown(f'<h2 class="section-heading">Analyse de similarité avec {embedding_type}</h2>', unsafe_allow_html=True)
        embed_sentences = preprocess_for_embedding(sentences)
        
        if embedding_type == "Word2Vec":
            model = train_word2vec(embed_sentences)
        else:
            model = train_fasttext(embed_sentences)

        sentence_vectors = np.array([
            get_sentence_vector(model, s.lower().split())
            for s in sentences
        ])
        similarities_embed = cosine_similarity(sentence_vectors)
        similarity_df_embed = pd.DataFrame(
            similarities_embed,
            columns=[f'Doc {i+1}' for i in range(len(sentences))],
            index=[f'Doc {i+1}' for i in range(len(sentences))]
        )
        st.write(f"Matrice de similarité ({embedding_type}) :")
        st.dataframe(similarity_df_embed)

        st.write(f"Heatmap des similarités ({embedding_type}) :")
        import plotly.express as px
        fig_heat = px.imshow(
            similarities_embed,
            labels=dict(x="Document", y="Document", color="Similarité")
        )
        st.plotly_chart(fig_heat)

        st.subheader("Explorer les mots similaires")
        word_to_explore = st.text_input("Entrez un mot pour voir ses plus proches voisins :")
        if word_to_explore:
            try:
                similar_words = model.wv.most_similar(word_to_explore.lower())
                st.write("Mots les plus similaires :")
                for word, score in similar_words:
                    st.write(f"- {word}: {score:.4f}")
            except KeyError:
                st.warning("Ce mot n'est pas dans le vocabulaire du modèle.")

    st.markdown("---")
    st.markdown("### 📝 Calculer la similarité d'une phrase avec tous les documents :")

    def calculer_similarite_local(phrase, documents):
        vect = TfidfVectorizer()
        corpus_local = [phrase] + documents
        tfidf_matrix_local = vect.fit_transform(corpus_local)
        sims_local = (tfidf_matrix_local * tfidf_matrix_local.T).A[0][1:]
        return sims_local

    doc_requete = st.number_input("Entrez le numéro du document (1 à N) :", min_value=1, max_value=len(sentences), step=1) - 1
    k_docs = st.slider("Choisissez le nombre de documents similaires à afficher :", 1, len(sentences)-1, 3)

    phrase_recherche = st.text_input("Entrez une phrase pour la comparer aux documents :")
    if phrase_recherche:
        sims_req = calculer_similarite_local(phrase_recherche, sentences)
        indices_sorted = sorted(range(len(sims_req)), key=lambda i: sims_req[i], reverse=True)
        st.write(f"Les {k_docs} documents les plus similaires à la phrase :")
        for idx in indices_sorted[:k_docs]:
            st.write(f"- Document {idx + 1} (sim={sims_req[idx]:.4f}) : {sentences[idx][:200]}...")

    if st.button("Trouver les documents similaires entre eux"):
        k_plus_proches = K_plus_proches_documents(doc_requete, k_docs, similarity_matrix, sentences)
        st.write(f"Les {k_docs} documents les plus similaires au document {doc_requete + 1} :")
        for idx2, sim2, phrase2 in k_plus_proches:
            st.write(f"- Document {idx2 + 1} (sim={sim2:.4f}) : {phrase2[:200]}...")



        # TF-IDF manuel
        st.markdown("### TF-IDF manuel (par Documents) :")

        def TF_IDF_New(liste_mots_differents_corpus, document, df):
            tf_doc = pd.Series(document).value_counts(normalize=True)
            tfidf_new_vector = pd.Series(0, index=liste_mots_differents_corpus)
            for mot in document:
                if mot in liste_mots_differents_corpus:
                    idf = np.log10(len(df) / (1 + df.get(mot, 0)))
                    tfidf_new_vector[mot] = tf_doc.get(mot, 0) * idf
            return tfidf_new_vector

        def create_matrices_tf(sentences_local):
            vect_bin = CountVectorizer(binary=True)
            vect_occ = CountVectorizer()
            X_bin = vect_bin.fit_transform(sentences_local)
            X_occ = vect_occ.fit_transform(sentences_local)
            terms_local = vect_occ.get_feature_names_out()

            tf_binary = pd.DataFrame(X_bin.toarray(), columns=terms_local, index=[f'Doc {i+1}' for i in range(len(sentences_local))])
            tf_occ = pd.DataFrame(X_occ.toarray(), columns=terms_local, index=[f'Doc {i+1}' for i in range(len(sentences_local))])
            tf_occ_normalized = tf_occ.div(tf_occ.sum(axis=1), axis=0)
            return tf_binary, tf_occ, tf_occ_normalized, terms_local

        def calculate_tfidf(tf_occ_local, num_documents_local):
            df_local = (tf_occ_local > 0).sum(axis=0)
            idf_local = np.log10(num_documents_local / df_local)
            tfidf_binary_local = tf_binary * idf_local.values
            tfidf_occ_local = tf_occ * idf_local.values
            tfidf_occ_normalized_local = tf_occ_normalized * idf_local.values
            return tfidf_binary_local, tfidf_occ_local, tfidf_occ_normalized_local

        sentences2 = split_into_sentences(chiraq_text)
        tf_binary, tf_occ, tf_occ_normalized, terms = create_matrices_tf(sentences2)
        tfidf_binary, tfidf_occ, tfidf_occ_normalized = calculate_tfidf(tf_occ, len(sentences2))

        st.write(f"Le texte contient {len(sentences2)} phrases.")
        st.write("**Matrice TF-IDF Binaire :**")
        st.dataframe(tfidf_binary)

        st.write("**Matrice TF-IDF Occurrence :**")
        st.dataframe(tfidf_occ)

        st.write("**Matrice TF-IDF Occurrence Normalisée :**")
        st.dataframe(tfidf_occ_normalized)

        # Distances sur ces TF-IDF
        distance_l1_binary = cdist(tfidf_binary, tfidf_binary, metric='cityblock')
        distance_l1_occ = cdist(tfidf_occ, tfidf_occ, metric='cityblock')
        distance_l1_occ_normalized = cdist(tfidf_occ_normalized, tfidf_occ_normalized, metric='cityblock')

        distance_l2_binary = cdist(tfidf_binary, tfidf_binary, metric='euclidean')
        distance_l2_occ = cdist(tfidf_occ, tfidf_occ, metric='euclidean')
        distance_l2_occ_normalized = cdist(tfidf_occ_normalized, tfidf_occ_normalized, metric='euclidean')

        distance_bray_curtis_binary = cdist(tfidf_binary, tfidf_binary, metric='braycurtis')
        distance_bray_curtis_occ = cdist(tfidf_occ, tfidf_occ, metric='braycurtis')
        distance_bray_curtis_occ_normalized = cdist(tfidf_occ_normalized, tfidf_occ_normalized, metric='braycurtis')

        st.markdown('<h2 class="section-heading">Distances L1</h2>', unsafe_allow_html=True)
        st.markdown("**Distance L1 Binaire :**")
        st.dataframe(distance_l1_binary)

        st.markdown("**Distance L1 Occurrence :**")
        st.dataframe(distance_l1_occ)

        st.markdown("**Distance L1 Occurrence Normalisée :**")
        st.dataframe(distance_l1_occ_normalized)

        st.markdown('<h2 class="section-heading">Distances L2 (Euclidienne)</h2>', unsafe_allow_html=True)
        st.markdown("**Distance L2 (Euclidienne) Binaire :**")
        st.dataframe(distance_l2_binary)

        st.markdown("**Distance L2 (Euclidienne) Occurrence :**")
        st.dataframe(distance_l2_occ)

        st.markdown("**Distance L2 (Euclidienne) Occurrence Normalisée :**")
        st.dataframe(distance_l2_occ_normalized)

        st.markdown('<h2 class="section-heading">Distance Bray-Curtis</h2>', unsafe_allow_html=True)
        st.markdown("**Distance Bray-Curtis Binaire :**")
        st.dataframe(distance_bray_curtis_binary)

        st.markdown("**Distance Bray-Curtis Occurrence :**")
        st.dataframe(distance_bray_curtis_occ)

        st.markdown("**Distance Bray-Curtis Occurrence Normalisée :**")
        st.dataframe(distance_bray_curtis_occ_normalized)

        st.markdown("---")
        st.markdown("*Fin de l'analyse.*")

try:
    plt.close('all')
except:
    pass
