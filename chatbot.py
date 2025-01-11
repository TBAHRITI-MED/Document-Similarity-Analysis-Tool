import streamlit as st
import chardet
import nltk
nltk.download("punkt")  # Télécharge le tokenizer de phrases si pas déjà fait
from nltk.tokenize import sent_tokenize
import numpy as np

# Pour l'approche basique
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pour l'approche avancée Sentence-BERT
from sentence_transformers import SentenceTransformer, util

# Pour l'approche avancée QA
from transformers import pipeline, CamembertTokenizer

##############################################
# 1) Chargement / cache des modèles
##############################################
@st.cache_resource
def load_retrieval_model():
    """
    Modèle Sentence-BERT multilingue pour la recherche sémantique.
    """
    return SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

@st.cache_resource
def load_qa_pipeline():
    """
    Pipeline QA francophone (CamemBERT fine-tuné sur FQuAD / Piaf),
    en forçant l'usage du tokenizer "lent".
    """
    my_tokenizer = CamembertTokenizer.from_pretrained(
        "etalab-ia/camembert-base-squadFR-fquad-piaf",
        use_fast=False,           
        trust_remote_code=True
    )
    my_pipeline = pipeline(
        "question-answering",
        model="etalab-ia/camembert-base-squadFR-fquad-piaf",
        tokenizer=my_tokenizer
    )
    return my_pipeline

# On charge nos deux modèles (dans le cache Streamlit)
retrieval_model = load_retrieval_model()
qa_pipeline = load_qa_pipeline()

###########################################
# 2) Interface Streamlit
###########################################
st.title("Chatbot : Approche Basique & Avancée")

# Choix de l'approche
approach = st.selectbox(
    "Choisissez l'approche que vous souhaitez utiliser :",
    ["Basique (Scénario prof)", "Avancée (Hugging Face QA)"]
)

uploaded_file = st.file_uploader("Téléchargez un fichier .txt", type="txt")

if uploaded_file:
    # 2a) Lire le fichier + détection d'encodage
    raw_data = uploaded_file.read()
    detect_result = chardet.detect(raw_data)
    guessed_encoding = detect_result["encoding"] if detect_result["encoding"] else "utf-8"
    document = raw_data.decode(guessed_encoding, errors="replace")

    st.write(f"**Encodage détecté** : {guessed_encoding}")

    # 2b) Découper le document en phrases
    sentences = sent_tokenize(document)

    st.write("### Phrases détectées dans le texte :")
    for idx, sentence in enumerate(sentences, start=1):
        st.write(f"{idx}. {sentence}")

    ###########################################
    # 3) Approche BASIQUE (scénario prof)
    ###########################################
    if approach == "Basique (Scénario prof)":
        st.subheader("Scénario Basique : TF-IDF, similarité cosinus, mot significatif")

        user_input = st.text_input("Entrez votre question :")

        if user_input:
            # Formules de politesse selon le début de la question
            polite_prefix = ""
            question_lower = user_input.lower().strip()
            if question_lower.startswith("comment"):
                polite_prefix = "Après analyse, "
            elif question_lower.startswith("pourquoi"):
                polite_prefix = "Car, "
            elif question_lower.startswith("peux-tu"):
                polite_prefix = "Oui, bien sûr! "

            # Vectorisation TF-IDF (question + corpus)
            corpus = [user_input] + sentences
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Séparation : question (ligne 0), corpus (ligne 1..)
            question_tfidf = tfidf_matrix[0:1]
            corpus_tfidf = tfidf_matrix[1:]

            # Similarités cosinus
            similarities = cosine_similarity(question_tfidf, corpus_tfidf).flatten()

            # Phrase la plus similaire
            best_index = np.argmax(similarities)
            best_sentence = sentences[best_index]

            # Mot le plus significatif dans la question (max TF-IDF)
            question_vec = question_tfidf.toarray().flatten()
            feature_names = vectorizer.get_feature_names_out()

            if np.sum(question_vec) != 0:
                max_idx = np.argmax(question_vec)
                most_significant_token = feature_names[max_idx]
            else:
                most_significant_token = ""

            # Vérifier la présence du token dans la phrase
            st.write("**Réponse :**")
            if most_significant_token:
                token_position = best_sentence.lower().find(most_significant_token.lower())
                if token_position != -1:
                    st.write(polite_prefix + best_sentence)
                else:
                    st.write(polite_prefix + best_sentence)
            else:
                st.write(polite_prefix + best_sentence)

            # Affichage du score
            st.write("---")
            st.write(f"Phrase la plus pertinente (score={similarities[best_index]:.4f}) : {best_sentence}")

    ###########################################
    # 4) Approche AVANCÉE (améliorée)
    ###########################################
    else:
        st.subheader("Scénario Avancé : Sentence-BERT + QA CamemBERT")

        # Choix du nombre de phrases à combiner
        k = st.slider("Combien de phrases combiner pour le contexte ?", 1, 10, 3)

        user_question = st.text_input("Posez votre question :")

        if user_question:
            st.write("Calcul de la similarité sémantique via Sentence-BERT...")

            # Encoder toutes les phrases du corpus
            sentence_embeddings = retrieval_model.encode(sentences, convert_to_tensor=True)
            # Encoder la question
            question_embedding = retrieval_model.encode(user_question, convert_to_tensor=True)

            # Similarités cosinus
            similarities = util.cos_sim(question_embedding, sentence_embeddings)[0].cpu().numpy()

            # Au lieu d'une seule phrase, on récupère les indices triés par ordre décroissant
            sorted_indices = np.argsort(similarities)[::-1]

            # Concaténer les k phrases les plus pertinentes en un seul bloc
            top_k_sentences = [sentences[i] for i in sorted_indices[:k]]
            context = " ".join(top_k_sentences)

            st.write("**Phrases sélectionnées :**")
            for i, idx in enumerate(sorted_indices[:k], start=1):
                st.write(f"{i}) [score={similarities[idx]:.4f}] : {sentences[idx]}")

            # Passage au pipeline QA
            st.write("Appel au modèle Question-Answering (CamemBERT)...")
            qa_input = {
                "question": user_question,
                "context": context
            }
            result = qa_pipeline(qa_input)

            st.write("**Réponse générée** : ", result["answer"])
            st.write(f"(Confiance : {result['score']:.4f})")

else:
    st.info("Veuillez téléverser un fichier .txt pour commencer.")
