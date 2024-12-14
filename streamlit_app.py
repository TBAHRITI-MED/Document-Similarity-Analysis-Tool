import streamlit as st
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, braycurtis
from scipy.special import kl_div
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Analyse de similarité de documents", page_icon="📄", layout="wide")
# Fonction de prétraitement avec options
def preprocess_text(sentences, remove_stopwords, apply_stemming, selected_stemmer_name, langue):
    # Choisir la langue des stopwords
    stop_words = set(stopwords.words('french' if langue == "Français" else 'english'))
    
    # Choisir le stemmer en fonction de l'option sélectionnée
    if selected_stemmer_name == "Porter":
        stemmer = PorterStemmer()
    elif selected_stemmer_name == "Lancaster":
        stemmer = LancasterStemmer()
    elif selected_stemmer_name == "Snowball (English)":
        stemmer = SnowballStemmer("english")
    elif selected_stemmer_name == "Snowball (French)":
        stemmer = SnowballStemmer("french")
    else:
        stemmer = None  # Pas de stemming si l'option n'est pas valide
    
    processed_sentences = []
    
    for sentence in sentences:
        # Tokenisation de la phrase
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        
        # Supprimer les stop words si nécessaire
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        
        # Appliquer le stemming si demandé
        if apply_stemming and stemmer:
            tokens = [stemmer.stem(word) for word in tokens]
                
        # Recréer la phrase traitée
        processed_sentences.append(" ".join(tokens))
    
    # Extraire les tokens uniques du texte traité
    unique_tokens = set()
    for sentence in processed_sentences:
        unique_tokens.update(sentence.split())
    
    return sorted(unique_tokens), processed_sentences


# 1. Diviser le texte en phrases
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences

# 3. Créer la matrice binaire et la matrice d'occurrences avec normalisation
def create_matrices(sentences, unique_tokens, normalization_type):
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

    # Appliquer la normalisation choisie
    if normalization_type == "Probabilité":
        occurrence_matrix = occurrence_matrix / occurrence_matrix.sum(axis=1, keepdims=True)
    elif normalization_type == "L2":
        occurrence_matrix = normalize(occurrence_matrix, norm='l2')

    return binary_matrix, occurrence_matrix

# 4. Calculer la distance de Manhattan
def calculate_manhattan_distance(matrix):
    return squareform(pdist(matrix, metric='cityblock'))

# 5. Calculer la distance euclidienne
def calculate_euclidean_distance(matrix):
    return squareform(pdist(matrix, metric='euclidean'))

# 6. Calculer la distance de Jaccard
def calculate_jaccard_distance(binary_matrix):
    jaccard_distances = pdist(binary_matrix, metric='jaccard')
    return squareform(jaccard_distances)

# 7. Calculer la distance de Hamming
def calculate_hamming_distance(binary_matrix):
    hamming_distances = pdist(binary_matrix, metric='hamming')
    return squareform(hamming_distances)

# 8. Calculer la distance Bray-Curtis
def calculate_bray_curtis_distance(matrix):
    bray_curtis_distances = pdist(matrix, metric=braycurtis)
    return squareform(bray_curtis_distances)


def calculate_kl_divergence(p, q):
    return np.sum(kl_div(p, q))
# 9. Calculer la distance Kullback-Leibler
def calculate_kullback_leibler_distance(matrix):
    # Ajout de petites constantes pour éviter la division par zéro
    matrix = np.clip(matrix, 1e-10, None)  # Clipper les valeurs
    kl_distances = squareform(pdist(matrix, metric=lambda u, v: np.sum(kl_div(u, v))))
    return kl_distances

# 10. Calculer la distance de Cosinus
def calculate_cosine_distance(matrix):
    """Calcule la distance de Cosinus entre les documents."""
    return squareform(pdist(matrix, metric='cosine'))

# 10. Créer la matrice de similarité
def calculate_similarity_matrix(distance_matrix):
    max_distance = np.max(distance_matrix)
    return 1 - (distance_matrix / max_distance)

def K_plus_proches_documents(doc_requete, k, similarity_matrix, sentences):
    similarites = similarity_matrix[doc_requete]
    similarites_idx = [(i, similarites[i]) for i in range(len(similarites)) if i != doc_requete]
    similarites_idx.sort(key=lambda x: x[1], reverse=True)
    
    # Récupérer les k documents les plus similaires avec leurs phrases
    return [(idx, similarity, sentences[idx]) for idx, similarity in similarites_idx[:k]]


def generate_wordcloud(text, background_color='white'):
    wordcloud = WordCloud(
        background_color=background_color,
        width=800,
        height=400,
        max_words=150
    ).generate(text)
    
    # Créer une figure matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Convertir le plot en image pour Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Fermer la figure pour libérer la mémoire
    buf.seek(0)
    return buf

# Fonction pour prétraiter le texte pour Word2Vec/FastText
def preprocess_for_embedding(sentences):
    return [sentence.lower().split() for sentence in sentences]

# Fonction pour entraîner Word2Vec
def train_word2vec(processed_sentences):
    model = Word2Vec(sentences=processed_sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Fonction pour entraîner FastText
def train_fasttext(processed_sentences):
    model = FastText(sentences=processed_sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Fonction pour obtenir le vecteur moyen d'une phrase
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

# Titre de l'application
st.title("Analyse de similarité de documents")

# Barre latérale
st.sidebar.subheader("Paramètres de configuration")
# Lien pour naviguer vers la page de recherche
st.sidebar.write("### Navigation")
page_selection = st.sidebar.radio("Sélectionner une page", ["Page principale", "Recherche dans un document","Chatbot"])

if page_selection == "Recherche dans un document":
    # Ouvrir la page de recherche dans un document dans un autre onglet
    webbrowser.open("http://localhost:8502/")
    st.write("Redirection vers la page de recherche dans un document...")
elif page_selection == "Chatbot":
    # Redirection vers la page du chatbot
    st.write("Redirection vers le chatbot...")
    webbrowser.open("http://localhost:8503/") 
# Ajout d'une description
elif page_selection == "Page principale":

    st.sidebar.write("""
    **Instructions :** Sélectionnez les options ci-dessus pour configurer l'analyse de similarité des documents. 
""")
# Choix de la langue
langue = st.sidebar.radio("Choisissez la langue du texte :", ("Français", "Anglais"))

# Options de prétraitement
st.sidebar.subheader("Prétraitement du texte")
remove_stopwords = st.sidebar.checkbox("Supprimer les stop words", value=True)
apply_stemming = st.sidebar.checkbox("Appliquer le stemming", value=False)

# Liste des stemmers disponibles
stemmers = {
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball (English)": SnowballStemmer("english"),
    "Snowball (French)": SnowballStemmer("french")
}

# Choix du type de stemming
selected_stemmer_name = st.sidebar.selectbox(
    "Choisissez le type de stemming :", 
    list(stemmers.keys())
)
selected_stemmer = stemmers[selected_stemmer_name]

# Choix du descripteur et de la normalisation
descripteur = st.sidebar.selectbox("Choisissez le descripteur à utiliser :", 
                                    ["Binaire", "Occurrence"])

normalization_type = st.sidebar.selectbox("Choisissez la méthode de normalisation :", 
                                           ["Aucune", "Probabilité", "L2"])

# Choix de la métrique de distance
distance_type = st.sidebar.selectbox("Choisissez la métrique de distance :", 
                                     
                                      ["Manhattan", "Euclidienne", "Jaccard", "Hamming", "Bray-Curtis", "Kullback-Leibler", "Cosinus"])
 
st.sidebar.write("""
    **Vous pouvez entrer votre texte manuellement ou en téléchargeant un fichier .txt.**
""")

# Méthode d'entrée de texte
input_mode = st.sidebar.radio("Comment voulez-vous entrer le texte ?", 
                               ("Rédaction manuelle", "Déposer un fichier .txt"))


st.sidebar.markdown("---")
st.sidebar.subheader("Paramètres d'embedding")
embedding_type = st.sidebar.selectbox(
    "Choisissez le type d'embedding",
    ["Word2Vec", "FastText", "Aucun"]
)

# Ajout de la section nuage de mots
st.sidebar.markdown("---")
st.sidebar.subheader("Nuage de mots")
show_wordcloud = st.sidebar.checkbox("Afficher le nuage de mots")
wordcloud_bg_color = st.sidebar.color_picker("Couleur de fond du nuage de mots", "#ffffff")
wordcloud_max_words = st.sidebar.slider("Nombre maximum de mots", 50, 300, 150)

# Zone de texte pour la rédaction manuelle ou fichier
if input_mode == "Rédaction manuelle":
    chiraq_text = st.text_area("Rédigez ou collez votre texte ici :")
else:
    uploaded_file = st.file_uploader("Déposez votre fichier texte ici", type="txt")
    if uploaded_file is not None:
        chiraq_text = uploaded_file.read().decode("utf-8")
    else:
        chiraq_text = """La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
Mes chers compatriotes de métropole, d'outre-mer et de l'étranger,
Nous venons de vivre un temps de grave inquiétude pour la Nation.
Mais ce soir, dans un grand élan la France a réaffirmé son attachement aux valeurs de la République.
Je salue la France, fidèle à elle-même, fidèle à ses grands idéaux, fidèle à sa vocation universelle et humaniste.
Je salue la France qui, comme toujours dans les moments difficiles, sait se retrouver sur l'essentiel. Je salue les Françaises et les Français épris de solidarité et de liberté, soucieux de s'ouvrir à l'Europe et au monde, tournés vers l'avenir.
J'ai entendu et compris votre appel pour que la République vive, pour que la Nation se rassemble, pour que la politique change. Tout dans l'action qui sera conduite, devra répondre à cet appel et s'inspirer d'une exigence de service et d'écoute pour chaque Française et chaque Français.
Ce soir, je veux vous dire aussi mon émotion et le sentiment que j'ai de la responsabilité qui m'incombe.
Votre choix d'aujourd'hui est un choix fondateur, un choix qui renouvelle notre pacte républicain. Ce choix m'oblige comme il oblige chaque responsable de notre pays. Chacun mesure bien, à l'aune de notre histoire, la force de ce moment exceptionnel.
Votre décision, vous l'avez prise en conscience, en dépassant les clivages traditionnels, et, pour certains d'entre vous, en allant au-delà même de vos préférences personnelles ou politiques.
La confiance que vous venez de me témoigner, je veux y répondre en m'engageant dans l'action avec détermination.
Président de tous les Français, je veux y répondre dans un esprit de rassemblement. Je veux mettre la République au service de tous. Je veux que les valeurs de liberté, d'égalité et de fraternité reprennent toute leur place dans la vie de chacune et de chacun d'entre nous.
La liberté, c'est la sécurité, la lutte contre la violence, le refus de l'impunité. Faire reculer l'insécurité est la première priorité de l'Etat pour les temps à venir.
La liberté, c'est aussi la reconnaissance du travail et du mérite, la réduction des charges et des impôts.
L'égalité, c'est le refus de toute discrimination, ce sont les mêmes droits et les mêmes devoirs pour tous.
La fraternité, c'est sauvegarder les retraites. C'est aider les familles à jouer pleinement leur rôle. C'est faire en sorte que personne n'éprouve plus le sentiment d'être laissé pour compte.
La France, forte de sa cohésion sociale et de son dynamisme économique, portera en Europe et dans le monde l'ambition de la paix, des libertés et de la solidarité.
Dans les prochains jours, je mettrai en place un gouvernement de mission, un gouvernement qui aura pour seule tâche de répondre à vos préoccupations et d'apporter des solutions à des problèmes trop longtemps négligés. Son premier devoir sera de rétablir l'autorité de l'Etat pour répondre à l'exigence de sécurité, et de mettre la France sur un nouveau chemin de croissance et d'emploi.
C'est par une action forte et déterminée, c'est par la solidarité de la Nation, c'est par l'efficacité des résultats obtenus, que nous pourrons lutter contre l'intolérance, faire reculer l'extrémisme, garantir la vitalité de notre démocratie. Cette exigence s'impose à chacun d'entre nous. Elle impliquera, au cours des prochaines années, vigilance et mobilisation de la part de tous.
Mes chers compatriotes,
Le mandat que vous m'avez confié, je l'exercerai dans un esprit d'ouverture et de concorde, avec pour exigence l'unité de la République, la cohésion de la Nation et le respect de l'autorité de l'Etat.
Les jours que nous venons de vivre ont ranimé la vigueur nationale, la vigueur de l'idéal démocratique français. Ils ont exprimé une autre idée de la politique, une autre idée de la citoyenneté.
Chacune et chacun d'entre vous, conscient de ses responsabilités, par un choix de liberté, a contribué, ce soir, à forger le destin de la France.
Il y a là un espoir qui ne demande qu'à grandir, un espoir que je veux servir.
Vive la République !
Vive la France !"""

if chiraq_text:
    with st.spinner("Traitement en cours..."):
        sentences = split_into_sentences(chiraq_text)
        unique_tokens, processed_sentences = preprocess_text(
        sentences, 
        remove_stopwords=remove_stopwords, 
        apply_stemming=apply_stemming, 
        selected_stemmer_name=selected_stemmer_name, 
        langue=langue)
        binary_matrix, occurrence_matrix = create_matrices(processed_sentences, unique_tokens, normalization_type)

    # Sélectionner la matrice en fonction du descripteur choisi
    if descripteur == "Binaire":
        matrix = binary_matrix
    else:
        matrix = occurrence_matrix

    # Calcul de la distance selon le choix de l'utilisateur
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
    
    # Afficher la matrice de distance sous forme de DataFrame
    distance_df = pd.DataFrame(distance_matrix, 
                               columns=[f'Doc {i+1}' for i in range(len(sentences))],
                               index=[f'Doc {i+1}' for i in range(len(sentences))])
    st.write("Matrice de distance :")
    st.dataframe(distance_df)

    # Calculer et afficher la matrice de similarité
    similarity_matrix = calculate_similarity_matrix(distance_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, 
                                 columns=[f'Doc {i+1}' for i in range(len(sentences))],
                                 index=[f'Doc {i+1}' for i in range(len(sentences))])
    st.write("Matrice de similarité :")
    st.dataframe(similarity_df)
    
    # Choisir un document pour trouver les plus proches
    # Affichage des documents avec des extraits de phrases
     
    options_docs = [
    f"Document {i + 1}: {processed_sentences[i][:100]}..." if len(processed_sentences[i]) > 100 else f"Document {i + 1}: {processed_sentences[i]}"
    for i in range(len(processed_sentences))
]
    st.write(options_docs) 
    


# Ajout dans la barre latérale

if chiraq_text:
    # Section pour le nuage de mots
    if show_wordcloud:
        st.subheader("Nuage de mots")
        wordcloud_buffer = generate_wordcloud(chiraq_text, background_color=wordcloud_bg_color)
        st.image(wordcloud_buffer)

    # Section pour l'embedding
    if embedding_type != "Aucun":
        st.title(f"Analyse de similarité avec {embedding_type}")
        
        # Prétraitement pour l'embedding
        processed_sentences = preprocess_for_embedding(sentences)
        
        # Entraînement du modèle selon le type choisi
        if embedding_type == "Word2Vec":
            model = train_word2vec(processed_sentences)
        else:  # FastText
            model = train_fasttext(processed_sentences)
        
        # Création des vecteurs de phrases
        sentence_vectors = np.array([
            get_sentence_vector(model, sentence.lower().split())
            for sentence in sentences
        ])
        
        # Calcul des similarités cosinus entre les phrases
        similarities = cosine_similarity(sentence_vectors)
        
        # Affichage de la matrice de similarité
        similarity_df = pd.DataFrame(
            similarities,
            columns=[f'Doc {i+1}' for i in range(len(sentences))],
            index=[f'Doc {i+1}' for i in range(len(sentences))]
        )
        
        st.write(f"Matrice de similarité ({embedding_type}):")
        st.dataframe(similarity_df)
        
        # Visualisation des similarités avec une heatmap
        st.write(f"Heatmap des similarités ({embedding_type}):")
        fig = px.imshow(
            similarities,
            labels=dict(x="Document", y="Document", color="Similarité"),
        
        )
        st.plotly_chart(fig)

        # Ajout d'une section pour explorer les mots similaires
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

try:
    # Gestion des erreurs pour l'affichage des figures
    plt.close('all')  # Fermer toutes les figures matplotlib ouvertes
except:
    pass
###################

def calculer_similarite(phrase, documents):
    vectorizer = TfidfVectorizer()
    # Fusionner la phrase recherchée et les documents
    corpus = [phrase] + documents
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calcul de la similarité entre la phrase et chaque document
    similarites = (tfidf_matrix * tfidf_matrix.T).A[0][1:]
    return similarites
##########
# Entrée pour le numéro de document
sentences = split_into_sentences(chiraq_text)

# Then create the number input without the sentences parameter
doc_requete = st.number_input("Entrez le numéro du document (1 à N) :", 
                             min_value=1, 
                             max_value=len(sentences), 
                             step=1) - 1

k = st.slider("Choisissez le nombre de documents similaires à afficher :", 
              1, 
              len(sentences)-1, 
              3)
# Entrée pour rechercher une phrase dans les documents
phrase_recherche = st.text_input("Entrez une phrase pour rechercher dans les documents :")

# Calculer les documents les plus similaires si la phrase de recherche est remplie
if phrase_recherche:
    similarites_recherche = calculer_similarite(phrase_recherche, sentences)
    
    # Trier les documents par similarité
    indices_similaires = sorted(range(len(similarites_recherche)), key=lambda i: similarites_recherche[i], reverse=True)
    
    st.write(f"Les {k} documents les plus similaires à la phrase recherchée :")
    for idx in indices_similaires[:k]:
        st.write(f"Document {idx + 1} avec similarité de {similarites_recherche[idx]:.4f} : {sentences[idx][:200]}...")
# Calculer les documents les plus similaires
if st.button("Trouver les documents similaires"):
    k_plus_proches = K_plus_proches_documents(doc_requete, k, similarity_matrix, sentences)
    st.write(f"Les {k} documents les plus similaires au document {doc_requete + 1} :")
    for idx, sim, phrase in k_plus_proches:
        st.write(f"Document {idx + 1} avec similarité de {sim:.4f} : {phrase[:200]}...")  # Afficher un extrait de 200 caractères de la phrase

    



# Fonction pour calculer TF-IDF_New pour un document
def TF_IDF_New(liste_mots_differents_corpus, document, df):
    """
    Cette fonction retourne un vecteur caractéristique du document de taille nb_mots_differents_corpus,
    où chaque bin contient le score TF-IDF_new.
    - liste_mots_differents_corpus : liste des mots uniques du corpus
    - document : liste de mots du document à analyser
    - df : Série ou dictionnaire donnant le nombre de documents contenant chaque mot (Document Frequency)
    """
    # Créer une série pour le calcul de la fréquence des mots dans le document
    tf_doc = pd.Series(document).value_counts(normalize=True)

    # Initialiser le vecteur TF-IDF_new avec des zéros pour chaque mot unique du corpus
    tfidf_new_vector = pd.Series(0, index=liste_mots_differents_corpus)

    # Calculer le TF-IDF pour chaque mot du document
    for mot in document:
        if mot in liste_mots_differents_corpus:
            # IDF est calculé comme le log du ratio entre le nombre total de documents et le nombre de documents contenant le mot
            idf = np.log10(len(df) / (1 + df.get(mot, 0)))
            tfidf_new_vector[mot] = tf_doc.get(mot, 0) * idf

    return tfidf_new_vector


# Fonction pour créer des matrices de fréquence
def create_matrices(sentences):
    vectorizer_binary = CountVectorizer(binary=True)
    vectorizer_occurrence = CountVectorizer()
    
    X_binary = vectorizer_binary.fit_transform(sentences)
    X_occ = vectorizer_occurrence.fit_transform(sentences)

    terms = vectorizer_occurrence.get_feature_names_out()

    # TF binaire
    tf_binary = pd.DataFrame(X_binary.toarray(), columns=terms, index=[f'Doc {i+1}' for i in range(len(sentences))])

    # TF occurrence
    tf_occ = pd.DataFrame(X_occ.toarray(), columns=terms, index=[f'Doc {i+1}' for i in range(len(sentences))])

    # TF occurrence normalisé
    tf_occ_normalized = tf_occ.div(tf_occ.sum(axis=1), axis=0)

    return tf_binary, tf_occ, tf_occ_normalized, terms

# Fonction pour calculer le TF-IDF manuellement
def calculate_tfidf(tf_occ, num_documents):
    # DF (Document Frequency)
    df = (tf_occ > 0).sum(axis=0)

    # IDF (Inverse Document Frequency)
    idf = np.log10(num_documents / df)

    # Calculer les matrices TF-IDF
    tfidf_binary = tf_binary * idf.values
    tfidf_occ = tf_occ * idf.values
    tfidf_occ_normalized = tf_occ_normalized * idf.values

    return tfidf_binary, tfidf_occ, tfidf_occ_normalized

# Interface utilisateur Streamlit
st.title("Analyse de similarité de documents avec TF-IDF")


if chiraq_text:
    sentences = split_into_sentences(chiraq_text)
    st.write(f"Le texte contient {len(sentences)} phrases.")

    unique_tokens, processed_sentences = preprocess_text(
        sentences, 
        remove_stopwords=remove_stopwords, 
        apply_stemming=apply_stemming, 
        selected_stemmer_name=selected_stemmer_name, 
        langue=langue)
    tf_binary, tf_occ, tf_occ_normalized, terms = create_matrices(sentences)

    # Calcul du TF-IDF
    tfidf_binary, tfidf_occ, tfidf_occ_normalized = calculate_tfidf(tf_occ, len(sentences))
    
    
    
################################################################################################
    # Calcul du DF (Document Frequency) pour chaque mot
    df = (tf_occ > 0).sum(axis=0)
    sentences = split_into_sentences(chiraq_text)
    # Interface utilisateur pour sélectionner une phrase/document
    selected_sentence = st.selectbox(
     "Sélectionnez une phrase pour calculer le vecteur TF-IDF_New :",
     options=sentences,
     format_func=lambda x: x[:100] + "..." if len(x) > 100 else x  # Afficher une partie de la phrase si elle est longue
    )

# Calcul du TF-IDF_New pour la phrase sélectionnée
    if selected_sentence:
     tfidf_new_vector = TF_IDF_New(terms, selected_sentence.split(), df)
     st.write("Vecteur TF-IDF_New pour la phrase sélectionnée :")
     st.dataframe(tfidf_new_vector)

################################################################################################


    # Afficher les matrices TF-IDF
    st.write("**Matrice TF-IDF Binaire :**")
    st.dataframe(tfidf_binary)

    st.write("**Matrice TF-IDF Occurrence :**")
    st.dataframe(tfidf_occ)

    st.write("**Matrice TF-IDF Occurrence Normalisée :**")
    st.dataframe(tfidf_occ_normalized)

    # Calcul des distances
    distance_l1_binary = cdist(tfidf_binary, tfidf_binary, metric='cityblock')
    distance_l1_occ = cdist(tfidf_occ, tfidf_occ, metric='cityblock')
    distance_l1_occ_normalized = cdist(tfidf_occ_normalized, tfidf_occ_normalized, metric='cityblock')

    distance_l2_binary = cdist(tfidf_binary, tfidf_binary, metric='euclidean')
    distance_l2_occ = cdist(tfidf_occ, tfidf_occ, metric='euclidean')
    distance_l2_occ_normalized = cdist(tfidf_occ_normalized, tfidf_occ_normalized, metric='euclidean')

    distance_bray_curtis_binary = cdist(tfidf_binary, tfidf_binary, metric='braycurtis')
    distance_bray_curtis_occ = cdist(tfidf_occ, tfidf_occ, metric='braycurtis')
    distance_bray_curtis_occ_normalized = cdist(tfidf_occ_normalized, tfidf_occ_normalized, metric='braycurtis')

    # Afficher les distances L1
    st.write("****Distance L1 Binaire :****")
    st.dataframe(distance_l1_binary)

    st.write("**Distance L1 Occurrence :**")
    st.dataframe(distance_l1_occ)

    st.write("**Distance L1 Occurrence Normalisée **:")
    st.dataframe(distance_l1_occ_normalized)

    # Afficher les distances L2
    st.write("**Distance L2 (Euclidienne) Binaire **:")
    st.dataframe(distance_l2_binary)

    st.write("**Distance L2 (Euclidienne) Occurrence **:")
    st.dataframe(distance_l2_occ)

    st.write("**Distance L2 (Euclidienne) Occurrence Normalisée **:")
    st.dataframe(distance_l2_occ_normalized)

    # Afficher les distances Bray-Curtis
    st.write("**Distance Bray-Curtis Binaire **:")
    st.dataframe(distance_bray_curtis_binary)

    st.write("**Distance Bray-Curtis Occurrence **:")
    st.dataframe(distance_bray_curtis_occ)

    st.write("**Distance Bray-Curtis Occurrence Normalisée :**")
    st.dataframe(distance_bray_curtis_occ_normalized)